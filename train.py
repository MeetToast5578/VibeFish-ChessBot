"""Train VibeFish on encoded shards, for a fixed wall-clock budget (cosine LR over time).

  python train.py --mix "data/L_*.npz:0.7,data/S_*.npz:0.3" --minutes 240 --out runs/main
  python train.py --mix ... --init runs/main/model.pt --lr 3e-4 --minutes 60 --out runs/ft   # fine-tune
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import ChessGPT, save, load
from data import load_shard
from teacher import to_grade

p = argparse.ArgumentParser()
p.add_argument("--mix", required=True, help="comma list of glob:weight[:kmax], kmax = max moves kept per position")
p.add_argument("--heldout", default="data/heldout.npz")
p.add_argument("--minutes", type=float, default=60)
p.add_argument("--batch", type=int, default=512)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--warmup", type=int, default=300, help="steps")
p.add_argument("--tau", type=float, default=0.03, help="softness of policy targets, in win-prob units")
p.add_argument("--w-action", type=float, default=1.0)
p.add_argument("--w-value", type=float, default=1.0)
p.add_argument("--d", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--init", help="start from this model file")
p.add_argument("--eval-every", type=int, default=500)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--compile", action="store_true", help="torch.compile the training forward (needs a working C++ compiler)")
p.add_argument("--multi-gpu", action="store_true", help="split each batch over all visible GPUs (nn.DataParallel)")
p.add_argument("--out", required=True)
args = p.parse_args()


class Source:
    """All shards matching a glob, sampled without replacement (reshuffled each pass). Shards stay
    separate (no big concatenate) so peak RAM is just the data (~530 B/position with kmax=8).
    shortcut: everything is held in RAM; switch to memmapped .npy shards past ~40M positions."""
    def __init__(self, pattern, weight, rng, kmax=None):
        # moves are stored best-first, so [:, :kmax] keeps the best ones; copy so the full arrays are freed
        self.parts = [(x, np.ascontiguousarray(i[:, :kmax]), np.ascontiguousarray(s[:, :kmax]))
                      for x, i, s in map(load_shard, sorted(glob.glob(pattern)))]
        assert self.parts, f"no shards match {pattern}"
        self.ends = np.cumsum([len(x) for x, _, _ in self.parts])
        self.n = int(self.ends[-1])
        self.weight, self.rng, self.pos, self.perm = weight, rng, 0, rng.permutation(self.n)
        self.name = pattern

    def take(self, n):
        """A list of (x, idx, score) pieces, one per shard hit; to_batch pads and joins them."""
        if self.pos + n > self.n:
            self.perm, self.pos = self.rng.permutation(self.n), 0
        ids = np.sort(self.perm[self.pos:self.pos + n])
        self.pos += n
        shard = np.searchsorted(self.ends, ids, side="right")
        out = []
        for s in np.unique(shard):
            local = ids[shard == s] - (self.ends[s - 1] if s else 0)
            x, i, sc = self.parts[s]
            out.append((x[local], i[local], sc[local]))
        return out


def to_batch(parts):
    k = max(i.shape[1] for _, i, _ in parts)
    pad = lambda a, v: np.pad(a, ((0, 0), (0, k - a.shape[1])), constant_values=v)
    x = torch.from_numpy(np.concatenate([p[0] for p in parts]))
    idx = torch.from_numpy(np.concatenate([pad(p[1], -1) for p in parts]).astype(np.int64))
    g = torch.from_numpy(to_grade(np.concatenate([pad(p[2], 0) for p in parts])).astype(np.float32))
    return x, idx, g


def losses(model, batch):
    x, idx, g = batch
    mask = idx >= 0
    gi = idx.clamp(min=0)
    policy, action, value = model(x)
    target = F.softmax((g / args.tau).masked_fill(~mask, -1e9), -1)
    logp = F.log_softmax(policy, -1).gather(1, gi)
    pol = -(target * logp).masked_fill(~mask, 0).sum(-1).mean()
    wp = g.clamp(0, 1)
    act = F.binary_cross_entropy_with_logits(action.gather(1, gi), wp, reduction="none")
    act = act.masked_fill(~mask, 0).sum() / mask.sum()
    best = wp.masked_fill(~mask, -1).max(-1).values
    v = value.float().clamp(1e-6, 1 - 1e-6)   # hand-written BCE: F.binary_cross_entropy refuses to run under CUDA autocast
    val = -(best * v.log() + (1 - best) * (1 - v).log()).mean()
    return pol, act, val, (policy, action, value)


@torch.no_grad()
def evaluate(model, batch):
    """Held-out positions have every legal move graded, so idx>=0 is exactly the legal set."""
    model.eval()
    x, idx, g = batch
    mask = idx >= 0
    pol, act, val, (policy, action, value) = losses(model, batch)
    wp = g.clamp(0, 1)
    best = wp.masked_fill(~mask, -1).max(-1).values
    out = {"pol": round(pol.item(), 4)}
    gi = idx.clamp(min=0)
    lp = F.log_softmax(policy.gather(1, gi).masked_fill(~mask, -1e9), -1)
    for name, s in [("policy", lp), ("mix", lp + F.logsigmoid(action.gather(1, gi)))]:
        pick = s.masked_fill(~mask, -1e9).argmax(-1, keepdim=True)
        chosen = wp.gather(1, pick).squeeze(1)
        out[f"{name}_top1"] = round((chosen >= best - 1e-9).float().mean().item(), 4)
        out[f"{name}_lost"] = round((best - chosen).mean().item(), 4)
    out["value_mae"] = round((value - best).abs().mean().item(), 4)
    model.train()
    return out


def main():
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    os.makedirs(args.out, exist_ok=True)
    sources = []
    for item in args.mix.split(","):
        pattern, w, *kmax = item.split(":")
        sources.append(Source(pattern, float(w), rng, int(kmax[0]) if kmax else None))
        print(f"{pattern}: {sources[-1].n} positions, weight {w}", flush=True)
    total_w = sum(s.weight for s in sources)
    counts = [round(args.batch * s.weight / total_w) for s in sources]
    hx, hi, hs = load_shard(args.heldout)
    dev = torch.device(args.device)
    # bf16 only with hardware support (Ampere+); T4 reports bf16 "supported" but emulates it ~10x slower
    amp = torch.bfloat16 if dev.type == "cuda" and torch.cuda.get_device_capability(dev) >= (8, 0) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and amp == torch.float16)
    held = [t.to(dev) for t in to_batch([(hx, hi, hs)])]

    model = (load(args.init).train() if args.init else ChessGPT(args.d, args.layers, args.heads)).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  cfg {model.cfg}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    log = open(f"{args.out}/log.jsonl", "a")
    rec = {"step": 0, "seen": 0, **evaluate(model, held)}
    print(rec, flush=True)
    log.write(json.dumps(rec) + "\n")

    # shares weights with `model`; compile + DataParallel crash together, so multi-GPU wins
    fast = torch.compile(model) if args.compile and not args.multi_gpu else model
    if args.multi_gpu and torch.cuda.device_count() > 1:
        fast = torch.nn.DataParallel(fast)
    t0, step, seen, run = time.time(), 0, 0, np.zeros(3)
    budget = args.minutes * 60
    while (frac := (time.time() - t0) / budget) < 1:
        lr = args.lr * min(1, (step + 1) / args.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for gr in opt.param_groups:
            gr["lr"] = lr
        batch = [t.to(dev, non_blocking=True) for t in to_batch([p for s, n in zip(sources, counts) if n for p in s.take(n)])]
        with torch.autocast("cuda", dtype=amp, enabled=dev.type == "cuda"):   # mixed precision on GPUs only (slower on this CPU)
            pol, act, val, _ = losses(fast, batch)
        loss = pol + args.w_action * act + args.w_value * val
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        seen += len(batch[0])
        run = 0.98 * run + 0.02 * np.array([pol.item(), act.item(), val.item()])
        if step % args.eval_every == 0:
            rec = {"step": step, "seen": seen, "min": round((time.time() - t0) / 60, 1), "lr": round(lr, 6),
                   "train_pol": round(float(run[0]) / (1 - 0.98 ** step), 4), **evaluate(model, held)}
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            save(model, f"{args.out}/model.pt", step=step)
    rec = {"step": step, "seen": seen, "min": round((time.time() - t0) / 60, 1), "final": True, **evaluate(model, held)}
    print(rec, flush=True)
    log.write(json.dumps(rec) + "\n")
    save(model, f"{args.out}/model.pt", step=step)


if __name__ == "__main__":
    main()

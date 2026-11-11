"""Train VibeFish on encoded shards, for a fixed wall-clock budget (cosine LR over time).

  python train.py --mix "data/L_*.npz:0.7,data/S_*.npz:0.3" --minutes 240 --out runs/main
  python train.py --mix ... --init runs/main/model.pt --lr 3e-4 --minutes 60 --out runs/ft   # fine-tune
"""
import argparse
import copy
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import ChessGPT, save, load, grow, with_cand
from data import load_shard
from teacher import to_grade

p = argparse.ArgumentParser()
p.add_argument("--mix", required=True, help="comma list of glob:weight[:kmax[:cand|act]], kmax = max moves kept per position; "
               "a trailing :cand marks a candidate-only source (ChessBench action-value rows: one graded move per position)")
p.add_argument("--cand-batch", type=int, default=512, help="rows per step drawn from the :cand sources")
p.add_argument("--act-batch", type=int, default=512, help="rows per step drawn from :act sources (one graded move per row: "
               "they train only the one-pass action-value head, never the policy or value)")
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
p.add_argument("--loops", type=int, default=1, help="passes through the layer stack (shared weights)")
p.add_argument("--attn-bias", action="store_true", help="learned square-pair attention bias per layer and head")
p.add_argument("--init", help="start from this model file")
p.add_argument("--grow", action="store_true", help="with --init: copy it into a model with --layers/--loops/--attn-bias (d, heads from --init)")
p.add_argument("--boost-from", type=int, help="layers from this index on (and the attention bias) train at --boost x lr")
p.add_argument("--boost", type=float, default=1.0, help="lr multiplier for --boost-from, e.g. for layers added by --grow")
p.add_argument("--ema", type=float, default=0.999, help="decay of the weight average saved as ema.pt (0 = off)")
p.add_argument("--opt-state", help="with --init: continue this saved optimizer state (opt.pt of the same model)")
p.add_argument("--cand", action="store_true", help="train candidate mode (one pass per legal move with it marked)")
p.add_argument("--cand-pos", type=int, default=256, help="positions per step used for candidate passes")
p.add_argument("--cand-k", type=int, default=4, help="moves per position: the best plus k-1 random graded ones")
p.add_argument("--w-cand", type=float, default=1.0, help="weight of the candidate win-prob BCE")
p.add_argument("--w-rank", type=float, default=1.0, help="weight of the candidate ranking loss (softmax over a position's k moves)")
p.add_argument("--w-plain", type=float, default=1.0, help="weight of the one-pass losses (policy, action, value); 0 skips that pass")
p.add_argument("--cand-lr-mult", type=float, default=10.0, help="lr multiplier for the new candidate parts (mark embedding, head)")
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
    known = mask & ~g.isnan()   # nan = best move without a score: a record's only move, so a one-hot policy target
    g = g.nan_to_num(0.0)
    target = F.softmax((g / args.tau).masked_fill(~mask, -1e9), -1)
    logp = F.log_softmax(policy, -1).gather(1, gi)
    pol = -(target * logp).masked_fill(~mask, 0).sum(-1).mean()
    wp = g.clamp(0, 1)
    act = F.binary_cross_entropy_with_logits(action.gather(1, gi), wp, reduction="none")
    act = act.masked_fill(~known, 0).sum() / known.sum().clamp(min=1)
    best = wp.masked_fill(~known, -1).max(-1).values
    has = known.any(-1)
    v = value.float().clamp(1e-6, 1 - 1e-6)   # hand-written BCE: F.binary_cross_entropy refuses to run under CUDA autocast
    val = (-(best * v.log() + (1 - best) * (1 - v).log()) * has).sum() / has.sum().clamp(min=1)
    return pol, act, val, (policy, action, value)


def cand_losses(model, batch, npos, k, min_known=2):
    """Candidate mode: for up to npos positions with >= min_known graded moves, the best move plus k-1
    random graded ones, each in its own pass. Returns (win-prob BCE, ranking CE over each position's
    moves; zero for positions with one move, e.g. ChessBench action-value rows)."""
    x, idx, g = batch
    known = (idx >= 0) & ~g.isnan()
    sel = torch.nonzero(known.sum(-1) >= min_known).squeeze(1)
    sel = sel[torch.randperm(len(sel), device=sel.device)[:npos]]   # rows come grouped by source: sample across all
    if len(sel) == 0:
        return torch.zeros((), device=x.device), torch.zeros((), device=x.device)
    kn = known[sel]
    r = torch.rand(kn.shape, device=x.device).masked_fill(~kn, -1.0)
    r[:, 0] = 2.0   # moves are stored best-first: always take the best
    cols = r.topk(min(k, kn.shape[1]), -1).indices
    valid = kn.gather(1, cols)
    mv = idx[sel].gather(1, cols).clamp(min=0)
    gr = g[sel].gather(1, cols).nan_to_num(0.0)
    logit = model(x[sel].repeat_interleave(cols.shape[1], 0), mv.flatten()).view(cols.shape).float()
    bce = F.binary_cross_entropy_with_logits(logit, gr.clamp(0, 1), reduction="none")
    bce = bce.masked_fill(~valid, 0).sum() / valid.sum().clamp(min=1)
    target = F.softmax((gr / args.tau).masked_fill(~valid, -1e9), -1)
    rank = -(target * F.log_softmax(logit.masked_fill(~valid, -1e9), -1)).masked_fill(~valid, 0).sum(-1).mean()
    return bce, rank


@torch.no_grad()
def cand_eval(model, batch, chunk=4096):
    """Candidate mode on held-out positions: every legal move scored in its own pass."""
    x, idx, g = batch
    mask = idx >= 0
    rows, cols = torch.nonzero(mask, as_tuple=True)
    logit = torch.full(idx.shape, -1e9, device=x.device)
    for a in range(0, len(rows), chunk):
        r, c = rows[a:a + chunk], cols[a:a + chunk]
        logit[r, c] = model(x[r], idx[r, c]).float()
    wp = g.clamp(0, 1)
    best = wp.masked_fill(~mask, -1).max(-1).values
    chosen = wp.gather(1, logit.argmax(-1, keepdim=True)).squeeze(1)
    return {"cand_top1": round((chosen >= best - 1e-9).float().mean().item(), 4),
            "cand_lost": round((best - chosen).mean().item(), 4)}


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
    if model.cfg.get("cand"):
        out.update(cand_eval(model, batch))
    model.train()
    return out


def ema_eval(ema, held):
    if ema is None:
        return {}
    r = evaluate(ema, held)
    return {f"ema_{k}": r[k] for k in ("policy_lost", "policy_top1", "cand_lost", "cand_top1") if k in r}


def checkpoint(model, ema, opt, step):
    save(model, f"{args.out}/model.pt", step=step)
    if ema is not None:
        save(ema, f"{args.out}/ema.pt", step=step)
    torch.save(opt.state_dict(), f"{args.out}/opt.pt")


def main():
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    os.makedirs(args.out, exist_ok=True)
    sources, cand_sources, act_sources = [], [], []
    for item in args.mix.split(","):
        pattern, w, *rest = item.split(":")
        src = Source(pattern, float(w), rng, int(rest[0]) if rest and rest[0] else None)
        tag = rest[1] if rest[1:] else ""
        {"cand": cand_sources, "act": act_sources}.get(tag, sources).append(src)
        print(f"{pattern}: {src.n} positions, weight {w}{f' ({tag} only)' if tag else ''}", flush=True)
    assert args.cand or not cand_sources, ":cand sources need --cand"
    split = lambda ss, n: [round(n * s.weight / sum(t.weight for t in ss)) for s in ss]
    counts, cand_counts, act_counts = split(sources, args.batch), split(cand_sources, args.cand_batch), split(act_sources, args.act_batch)
    hx, hi, hs = load_shard(args.heldout)
    dev = torch.device(args.device)
    # bf16 only with hardware support (Ampere+); T4 reports bf16 "supported" but emulates it ~10x slower
    amp = torch.bfloat16 if dev.type == "cuda" and torch.cuda.get_device_capability(dev) >= (8, 0) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and amp == torch.float16)
    held = [t.to(dev) for t in to_batch([(hx, hi, hs)])]

    if args.init:
        model = grow(load(args.init), args.layers, args.loops, args.attn_bias) if args.grow else load(args.init)
    else:
        model = ChessGPT(args.d, args.layers, args.heads, args.loops, args.attn_bias)
    if args.cand:
        model = with_cand(model)
    model = model.train().to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  cfg {model.cfg}", flush=True)
    fast_ids = set()
    if args.boost_from is not None:
        fast_ids = {id(t) for layer in model.body.layers[args.boost_from:] for t in layer.parameters()}
        if model.attn_bias is not None:
            fast_ids.add(id(model.attn_bias))
    cand_ids = {id(t) for n, t in model.named_parameters() if n.startswith("cand_")} if args.cand else set()
    groups = [{"params": [t for t in model.parameters() if id(t) not in fast_ids | cand_ids], "mult": 1.0},
              {"params": [t for t in model.parameters() if id(t) in fast_ids], "mult": args.boost},
              {"params": [t for t in model.parameters() if id(t) in cand_ids], "mult": args.cand_lr_mult}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    if args.opt_state:
        opt.load_state_dict(torch.load(args.opt_state, map_location=dev))
    ema = copy.deepcopy(model).requires_grad_(False) if args.ema else None
    log = open(f"{args.out}/log.jsonl", "a")
    rec = {"step": 0, "seen": 0, **evaluate(model, held)}
    print(rec, flush=True)
    log.write(json.dumps(rec) + "\n")

    # shares weights with `model`; compile + DataParallel crash together, so multi-GPU wins
    fast = torch.compile(model) if args.compile and not args.multi_gpu else model
    if args.multi_gpu and torch.cuda.device_count() > 1:
        fast = torch.nn.DataParallel(fast)
    t0, step, seen, run = time.time(), 0, 0, np.zeros(7)
    zero = torch.zeros((), device=dev)
    budget = args.minutes * 60
    while (frac := (time.time() - t0) / budget) < 1:
        lr = args.lr * min(1, (step + 1) / args.warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        for gr in opt.param_groups:
            gr["lr"] = lr * gr["mult"]
        batch = [t.to(dev, non_blocking=True) for t in to_batch([p for s, n in zip(sources, counts) if n for p in s.take(n)])]
        opt.zero_grad(set_to_none=True)

        def back(part):   # backward per part, so only one part's graph is in GPU memory at a time
            if part.requires_grad:
                scaler.scale(part).backward()

        amp_on = lambda: torch.autocast("cuda", dtype=amp, enabled=dev.type == "cuda")   # mixed precision on GPUs only
        with amp_on():
            pol, act, val, _ = losses(fast, batch) if args.w_plain else (zero, zero, zero, None)
        back(args.w_plain * (pol + args.w_action * act + args.w_value * val))
        with amp_on():
            cb, rk = cand_losses(fast, batch, args.cand_pos, args.cand_k) if args.cand else (zero, zero)
        back(args.w_cand * cb + args.w_rank * rk)
        av = zero
        if cand_sources:   # one graded move per row: candidate BCE only
            cbatch = [t.to(dev, non_blocking=True) for t in to_batch([p for s, n in zip(cand_sources, cand_counts) if n for p in s.take(n)])]
            with amp_on():
                av, _ = cand_losses(fast, cbatch, args.cand_batch, 1, min_known=1)
            back(args.w_cand * av)
        ac = zero
        if act_sources:   # one graded move per row: one-pass action-value BCE only
            abatch = [t.to(dev, non_blocking=True) for t in to_batch([p for s, n in zip(act_sources, act_counts) if n for p in s.take(n)])]
            with amp_on():
                _, ac, _, _ = losses(fast, abatch)
            back(args.w_action * ac)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - args.ema)
        step += 1
        seen += len(batch[0])
        run = 0.98 * run + 0.02 * np.array([pol.item(), act.item(), val.item(), cb.item(), rk.item(), av.item(), ac.item()])
        if step % args.eval_every == 0:
            tr = run / (1 - 0.98 ** step)
            rec = {"step": step, "seen": seen, "min": round((time.time() - t0) / 60, 1), "lr": round(lr, 6),
                   "train_pol": round(float(tr[0]), 4), **({"train_cand": round(float(tr[3]), 4), "train_rank": round(float(tr[4]), 4)} if args.cand else {}),
                   **({"train_av": round(float(tr[5]), 4)} if cand_sources else {}),
                   **({"train_act_av": round(float(tr[6]), 4)} if act_sources else {}),
                   **evaluate(model, held), **ema_eval(ema, held)}
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            checkpoint(model, ema, opt, step)
    rec = {"step": step, "seen": seen, "min": round((time.time() - t0) / 60, 1), "final": True, **evaluate(model, held), **ema_eval(ema, held)}
    print(rec, flush=True)
    log.write(json.dumps(rec) + "\n")
    checkpoint(model, ema, opt, step)


if __name__ == "__main__":
    main()

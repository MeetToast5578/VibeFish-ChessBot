"""Stockfish-graded training loop (DAgger-style distillation).

Round 0: Stockfish plays itself; every position is graded -> train.
Round r>0: the model plays (teacher steps in with prob beta). Stockfish grades every
position the *model* reached, so it learns to recover from its own mistakes.
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import time

import torch
import torch.nn.functional as F

import chess
from model import ChessGPT, encode
import teacher

p = argparse.ArgumentParser()
p.add_argument("--rounds", type=int, default=6)
p.add_argument("--games", type=int, default=200, help="games per round")
p.add_argument("--depth", type=int, default=8)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--tau", type=float, default=0.03, help="softness of targets, in win-prob units")
p.add_argument("--workers", type=int, default=os.cpu_count())
p.add_argument("--out", default="run")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
CKPT = f"{args.out}/model.pt"
torch.manual_seed(0)


def generate(model_path, n, beta, temperature, seed):
    jobs = [(beta, temperature, args.depth, 160, seed * 100000 + i) for i in range(n)]
    with mp.Pool(args.workers, teacher.init_worker, (model_path,)) as pool:
        res = pool.map(teacher.play_game, jobs, chunksize=1)
    samples = [s for r in res for s in r[0]]
    plies = sum(r[1] for r in res)
    lost = sum(r[2] for r in res)
    return samples, (lost / plies if plies else None)


def batchify(samples):
    x = torch.tensor([encode(chess.Board(s[0])) if isinstance(s[0], str) else s[0] for s in samples])
    legal = torch.zeros(len(samples), 4096, dtype=torch.bool)
    wp = torch.zeros(len(samples), 4096)
    best = torch.zeros(len(samples))
    for i, (_, g) in enumerate(samples):
        idx = torch.tensor(list(g.keys()))
        legal[i, idx] = True
        wp[i, idx] = torch.tensor(list(g.values()))
        best[i] = max(g.values())
    return x, legal, wp, best


def losses(model, batch):
    x, legal, wp, best = batch
    logits, v = model(x)
    logits = logits.masked_fill(~legal, -1e9)
    target = F.softmax((wp / args.tau).masked_fill(~legal, -1e9), -1)   # Stockfish's grades -> soft target
    policy = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
    value = F.mse_loss(v, best)
    return policy, value, logits, v


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    x, legal, wp, best = data
    _, _, logits, v = losses(model, data)
    pick = logits.argmax(-1)
    top1 = (pick == (wp.masked_fill(~legal, -1)).argmax(-1)).float().mean().item()
    lost = (best - wp.gather(1, pick[:, None]).squeeze(1)).mean().item()
    model.train()
    return {"top1": round(top1, 3), "wp_lost": round(lost, 4), "value_mae": round((v - best).abs().mean().item(), 4)}


def train(model, opt, samples):
    model.train()
    for ep in range(args.epochs):
        random.shuffle(samples)
        tot = n = 0
        for i in range(0, len(samples), 256):
            pol, val, _, _ = losses(model, batchify(samples[i:i + 256]))
            loss = pol + val
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += pol.item(); n += 1
        print(f"   epoch {ep + 1}: policy loss {tot / n:.3f}", flush=True)


if __name__ == "__main__":
    # fixed held-out set: positions from Stockfish games (never trained on)
    eval_path = f"{args.out}/eval.pkl"
    if not os.path.exists(eval_path):
        ev, _ = generate(None, 40, 1.0, 0, seed=999)
        ev = random.Random(0).sample(ev, min(2000, len(ev)))
        pickle.dump(ev, open(eval_path, "wb"))
    eval_data = batchify(pickle.load(open(eval_path, "rb")))

    model = ChessGPT()
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    data, log = [], open(f"{args.out}/log.jsonl", "a")
    print("untrained:", evaluate(model, eval_data))

    for r in range(args.rounds):
        t = time.time()
        beta = 1.0 if r == 0 else max(0.1, 0.5 - 0.1 * (r - 1))
        new, live_loss = generate(CKPT if r else None, args.games, beta, 0.3, seed=r)
        pickle.dump(new, open(f"{args.out}/data_round{r}.pkl", "wb"))   # FENs + grades, reusable later
        data += [(encode(chess.Board(f)), g) for f, g in new]          # encode once
        gen_s = time.time() - t
        print(f"round {r}: beta={beta:.1f} +{len(new)} positions (total {len(data)}) in {gen_s:.0f}s"
              + (f", model's own moves lost {live_loss:.3f} win-prob on average" if live_loss is not None else ""), flush=True)
        train(model, opt, data)
        torch.save(model.state_dict(), CKPT)
        m = evaluate(model, eval_data)
        rec = {"round": r, "beta": beta, "positions": len(data), "live_wp_lost": live_loss, **m,
               "minutes": round((time.time() - t) / 60, 1)}
        print("   eval:", m, flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()

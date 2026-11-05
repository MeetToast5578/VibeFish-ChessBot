"""Build training data. Every source becomes records (fen, moves pushed after fen, {uci: raw score}),
which are pickled (reusable if the encoding changes) and encoded into .npz shards:
  x [N, SEQ, NF] uint8   idx [N, K] int16 (move index, -1 pad)   score [N, K] int32 (raw score)

  python data.py lichess --src data/eval_part.zst --start 20000 --count 5000000 --out data/L
  python data.py sf      --src data/eval_part.zst --start 20000 --count 300000 --depth 8 --out data/S
  python data.py sf      --src data/eval_part.zst --start 0 --count 1000 --depth 12 --out data/heldout
  python data.py dagger  --model models/vibefish_v2.pt --games 2000 --depth 8 --out data/D1
"""
import argparse
import io
import itertools
import json
import multiprocessing as mp
import os
import pickle

import chess
import numpy as np

import teacher
from model import encode, orient_move, SEQ, NF

EVAL_LINES = 20_000   # the first lines of the Lichess eval file are reserved for held-out evaluation


def lichess_lines(path, start, count):
    import zstandard
    with open(path, "rb") as f:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8")
        try:
            yield from itertools.islice(text, start, start + count)
        except zstandard.ZstdError:   # a partial download ends mid-frame; keep what we got
            return


def parse_lichess(line):
    """Lichess eval DB line -> record. Uses the eval with the most PVs (deepest on ties)."""
    d = json.loads(line)
    fen = d["fen"]
    ev = max(d["evals"], key=lambda e: (len(e["pvs"]), e["depth"]))
    sign = 1 if fen.split()[1] == "w" else -1     # Lichess scores are from white's view
    scores = {}
    for pv in ev["pvs"]:
        move = pv["line"].split()[0]
        if "mate" in pv:
            m = pv["mate"] * sign
            s = teacher.MATE - m if m > 0 else -teacher.MATE - m
        else:
            s = pv["cp"] * sign
        scores[move] = s
    return fen, [], scores


def encode_record(rec, kmax=96):
    fen, moves, scores = rec
    board = chess.Board(fen)
    for m in moves:
        board.push_uci(m)
    best = {}
    for uci, s in scores.items():
        i = orient_move(board, chess.Move.from_uci(uci))
        best[i] = max(best.get(i, s), s)   # under-promotions fold into the queen promotion
    top = sorted(best.items(), key=lambda kv: -kv[1])[:kmax]
    return encode(board), top


_exclude = set()


def _init_exclude(path):
    global _exclude
    _exclude = set(open(path).read().split("\n")) if path else set()


def _encode_lines(lines):
    out = []
    for line in lines:
        try:
            rec = parse_lichess(line)
            if " ".join(rec[0].split()[:4]) not in _exclude:
                out.append(encode_record(rec))
        except (ValueError, KeyError, AssertionError):
            pass   # odd FEN / illegal PV move in the dump
    return out


def to_shard(encoded, path):
    k = max(len(t) for _, t in encoded)
    x = np.stack([e for e, _ in encoded])
    idx = np.full((len(encoded), k), -1, np.int16)
    score = np.zeros((len(encoded), k), np.int32)
    for r, (_, top) in enumerate(encoded):
        idx[r, :len(top)] = [i for i, _ in top]
        score[r, :len(top)] = [s for _, s in top]
    np.savez(path, x=x, idx=idx, score=score)
    print(f"wrote {path}: {len(x)} positions, k={k}", flush=True)


def load_shard(path):
    z = np.load(path)
    assert z["x"].shape[1:] == (SEQ, NF), f"{path} has an old encoding"
    return z["x"], z["idx"], z["score"]


def chunks(it, n):
    it = iter(it)
    while batch := list(itertools.islice(it, n)):
        yield batch


def cmd_lichess(a):
    shard, buf = 0, []
    with mp.Pool(a.workers, _init_exclude, (a.exclude,)) as pool:
        for enc in pool.imap(_encode_lines, chunks(lichess_lines(a.src, a.start, a.count), 2000)):
            buf += enc
            if len(buf) >= a.shard_size:
                to_shard(buf, f"{a.out}_{shard:02d}.npz"); shard += 1; buf = []
    if buf:
        to_shard(buf, f"{a.out}_{shard:02d}.npz")


def save_records(records, out):
    records = [r for r in records if r and r[2]]
    pickle.dump(records, open(f"{out}.pkl", "wb"))
    with mp.Pool(4) as pool:
        to_shard(pool.map(encode_record, records, chunksize=256), f"{out}.npz")


def cmd_sf(a):
    fens = [json.loads(l)["fen"] for l in lichess_lines(a.src, a.start, a.count)]
    with mp.Pool(a.workers, teacher.init_worker, (a.depth,)) as pool:
        records = pool.map(teacher.grade_fen, fens, chunksize=64)
    save_records(records, a.out)


def cmd_dagger(a):
    with mp.Pool(a.workers, teacher.init_worker, (a.depth, a.model)) as pool:
        res = pool.map(teacher.play_game, range(a.seed, a.seed + a.games), chunksize=1)
    lost = [l for _, ls in res for l in ls]
    print(f"model's own moves lost {np.mean(lost):.4f} win prob on average over {len(lost)} moves")
    save_records([r for recs, _ in res for r in recs], a.out)


def cmd_puzzles(a):
    """Fixed benchmark: a.count puzzles spread evenly over 7 rating bands (from the Lichess puzzle
    dump), plus bench/exclude.txt listing every position we must solve (kept out of training)."""
    import csv
    import zstandard
    quota = a.count // 7
    bands, rows = {}, []
    with open(a.src, "rb") as f:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8")
        for r in csv.DictReader(text):
            band = min(max(int(r["Rating"]), 400), 2800) // 400
            if bands.get(band, 0) < quota:
                bands[band] = bands.get(band, 0) + 1
                rows.append(r)
            if len(rows) >= quota * 7:
                break
    keep = ["PuzzleId", "FEN", "Moves", "Rating", "Themes"]
    with open("bench/puzzles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keep, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    fens = set()
    for r in rows:
        b = chess.Board(r["FEN"])
        for m in r["Moves"].split():
            b.push_uci(m)
            fens.add(" ".join(b.fen().split()[:4]))
    open("bench/exclude.txt", "w").write("\n".join(sorted(fens)))
    print(f"{len(rows)} puzzles, {len(fens)} excluded positions, bands {sorted(bands.items())}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["lichess", "sf", "dagger", "puzzles"])
    p.add_argument("--exclude", help="file of FENs (first 4 fields) to skip")
    p.add_argument("--src", default="data/eval_part.zst")
    p.add_argument("--start", type=int, default=EVAL_LINES)
    p.add_argument("--count", type=int, default=1_000_000)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--model")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out")
    a = p.parse_args()
    {"lichess": cmd_lichess, "sf": cmd_sf, "dagger": cmd_dagger, "puzzles": cmd_puzzles}[a.cmd](a)

"""Build training data. Every source becomes records (fen, moves pushed after fen, {uci: raw score}),
which are pickled (reusable if the encoding changes) and encoded into .npz shards:
  x [N, SEQ, NF] uint8   idx [N, K] int16 (move index, -1 pad)   score [N, K] int32 (raw score)

  python data.py lichess --src data/eval_part.zst --start 20000 --count 5000000 --out data/L
  python data.py sf      --src data/eval_part.zst --start 20000 --count 300000 --depth 8 --out data/S
  python data.py sf      --src data/eval_part.zst --start 0 --count 1000 --depth 12 --out data/heldout
  python data.py dagger  --model models/vibefish_v2.pt --games 2000 --depth 8 --out data/D1
  python data.py chessbench --src CB_BC --start 0 --count 20000000 --exclude bench/exclude.txt,bench/heldout_fens.txt --out data/CB
  python data.py dmav --src CB_AV --file 7 --start 0 --count 8000000 --exclude bench/exclude.txt,bench/heldout_fens.txt --out data/AV7   # candidate-mode only
"""
import argparse
import io
import itertools
import json
import multiprocessing as mp
import os
import pickle
import struct
import urllib.request

import chess
import numpy as np

import teacher
from model import encode, orient_move, SEQ, NF

EVAL_LINES = 20_000   # the first lines of the Lichess eval file are reserved for held-out evaluation
# ChessBench (DeepMind, CC-BY 4.0): 527M Lichess positions with Stockfish 16's best move (50 ms/move)
CB_BC = "https://storage.googleapis.com/searchless_chess/data/train/behavioral_cloning_data.bag"
# ChessBench action-value set: (position, move, win prob) for every legal move, shuffled over 2,148 files
CB_AV = "https://storage.googleapis.com/searchless_chess/data/train/action_value-{:05d}-of-02148_data.bag"


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
_kmax = 96


def _init_exclude(path, kmax=96):
    global _exclude, _kmax
    _exclude = set().union(*(open(f).read().split("\n") for f in path.split(","))) if path else set()
    _kmax = kmax


def _encode_lines(lines):
    out = []
    for line in lines:
        try:
            rec = parse_lichess(line)
            if " ".join(rec[0].split()[:4]) not in _exclude:
                out.append(encode_record(rec, _kmax))
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
    np.savez_compressed(path, x=x, idx=idx, score=score)
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
    with mp.Pool(a.workers, _init_exclude, (a.exclude, a.kmax)) as pool:
        for enc in pool.imap(_encode_lines, chunks(lichess_lines(a.src, a.start, a.count), 2000)):
            buf += enc
            if len(buf) >= a.shard_size:
                to_shard(buf, f"{a.out}_{shard:02d}.npz"); shard += 1; buf = []
    if buf:
        to_shard(buf, f"{a.out}_{shard:02d}.npz")


def read_range(src, a, b):
    """Bytes [a, b) of a local file or an http(s) URL (range request)."""
    if src.startswith("http"):
        req = urllib.request.Request(src, headers={"Range": f"bytes={a}-{b - 1}"})
        for attempt in range(5):
            try:
                data = urllib.request.urlopen(req, timeout=120).read()
                assert len(data) == b - a, "short read"
                return data
            except Exception:
                if attempt == 4:
                    raise
    with open(src, "rb") as f:
        f.seek(a)
        return f.read(b - a)


def bag_records(src, start, count, block=500_000):
    """Raw records start..start+count of a Bagz file (DeepMind's format: records back to back, then
    one little-endian int64 end offset per record, then the offset of that index)."""
    size = (int(urllib.request.urlopen(urllib.request.Request(src, method="HEAD")).headers["Content-Length"])
            if src.startswith("http") else os.path.getsize(src))
    (index,) = struct.unpack("<Q", read_range(src, size - 8, size))
    n = (size - 8 - index) // 8 + 1   # the last index entry doubles as the index offset
    for lo in range(start, min(start + count, n), block):
        hi = min(lo + block, start + count, n)
        ends = np.frombuffer(read_range(src, index + 8 * lo, index + 8 * hi), "<i8")
        first = struct.unpack("<q", read_range(src, index + 8 * (lo - 1), index + 8 * lo))[0] if lo else 0
        data = read_range(src, first, int(ends[-1]))
        starts = np.concatenate([[first], ends[:-1]]) - first
        for a, b in zip(starts, ends - first):
            yield data[a:b]


def parse_bc(rec):
    """ChessBench behavioral-cloning record (Beam tuple: varint length + utf-8 FEN, then the move as
    the rest of the record; the last field has no length) -> our record, best move's score unknown."""
    n = shift = i = 0
    while True:
        c = rec[i]; i += 1
        n |= (c & 127) << shift; shift += 7
        if c < 128:
            break
    return rec[i:i + n].decode(), [], {rec[i + n:].decode(): teacher.UNKNOWN}


def _encode_bc(recs):
    out = []
    for r in recs:
        rec = parse_bc(r)
        if " ".join(rec[0].split()[:4]) not in _exclude:
            out.append(encode_record(rec))
    return out


def parse_av(rec):
    """ChessBench action-value record (varint-length FEN, varint-length move, big-endian float64 win
    prob from Stockfish 16, same curve as ours) -> our record with that one move's centipawn score.
    DeepMind shuffled these across all 2,148 files, so a position here has ~1 graded move."""
    fields, i = [], 0
    for _ in range(2):
        n = shift = 0
        while True:
            c = rec[i]; i += 1
            n |= (c & 127) << shift; shift += 7
            if c < 128:
                break
        fields.append(rec[i:i + n].decode()); i += n
    wp = min(max(struct.unpack(">d", rec[i:i + 8])[0], 1e-6), 1 - 1e-6)
    return fields[0], [], {fields[1]: round(np.log(wp / (1 - wp)) / 0.00368208)}


def _encode_av(recs):
    return [encode_record(rec) for rec in map(parse_av, recs) if " ".join(rec[0].split()[:4]) not in _exclude]


def cmd_chessbench(a):
    shard, buf = 0, []
    av = a.cmd == "dmav"
    src = (CB_AV.format(a.file) if av else CB_BC) if a.src in ("CB_BC", "CB_AV") else a.src
    with mp.Pool(a.workers, _init_exclude, (a.exclude,)) as pool:
        for enc in pool.imap(_encode_av if av else _encode_bc, chunks(bag_records(src, a.start, a.count), 5000)):
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


def puzzle_fens(path, start, count, exclude):
    """Middlegame/tactical positions from the Lichess puzzle dump: the game position before the
    opponent's move, and the puzzle position after it. Benchmark positions are skipped."""
    import csv
    lines = lichess_lines(path, start + 1, count // 2)
    for r in csv.DictReader(itertools.chain(["PuzzleId,FEN,Moves,Rating"], lines)):
        b = chess.Board(r["FEN"])
        before = b.fen()
        b.push_uci(r["Moves"].split()[0])
        for fen in (before, b.fen()):
            if " ".join(fen.split()[:4]) not in exclude:
                yield fen


def cmd_sf(a):
    if "puzzle" in a.src:
        _init_exclude(a.exclude)
        fens = list(puzzle_fens(a.src, a.start, a.count, _exclude))
    else:
        fens = [json.loads(l)["fen"] for l in lichess_lines(a.src, a.start, a.count)]
    with mp.Pool(a.workers, teacher.init_worker, (a.depth,)) as pool:
        records = pool.map(teacher.grade_fen, fens, chunksize=64)
    save_records(records, a.out)


def cmd_dagger(a):
    starts = [l.split(",")[1] for l in itertools.islice(lichess_lines(a.start_puzzles, max(a.start, 1), a.games), a.games)] if a.start_puzzles else []
    with mp.Pool(a.workers, teacher.init_worker, (a.depth, a.model, a.strip_history, starts)) as pool:
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
    p.add_argument("cmd", choices=["lichess", "sf", "dagger", "puzzles", "chessbench", "dmav"])
    p.add_argument("--file", type=int, default=0, help="dmav: which of the 2,148 action-value files (with --src CB_AV)")
    p.add_argument("--exclude", help="comma list of files of FENs (first 4 fields) to skip")
    p.add_argument("--kmax", type=int, default=96, help="lichess: keep at most this many (best) moves per position")
    p.add_argument("--src", default="data/eval_part.zst")
    p.add_argument("--start", type=int, default=EVAL_LINES)
    p.add_argument("--count", type=int, default=1_000_000)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--model")
    p.add_argument("--strip-history", action="store_true", help="dagger: the model plays without history features")
    p.add_argument("--start-puzzles", help="dagger: start half the games from game positions in this puzzle dump (rows from --start on, past the benchmark rows)")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out")
    a = p.parse_args()
    {"lichess": cmd_lichess, "sf": cmd_sf, "dagger": cmd_dagger, "puzzles": cmd_puzzles,
     "chessbench": cmd_chessbench, "dmav": cmd_chessbench}[a.cmd](a)

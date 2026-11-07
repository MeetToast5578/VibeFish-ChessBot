"""Benchmarks for a searchless bot (one forward pass per move).

  python evaluate.py --model models/vibefish_v2.pt --puzzles --heldout --match 1320,1600 --games 40
  python evaluate.py --v1 models/v1 ...                 # VibeFish v1 (its own model.py + weights)
  python evaluate.py --model models/vibefish_v3.pt --vs models/vibefish_v2.pt --games 100   # head-to-head

puzzles  bench/puzzles.csv: 1,050 Lichess puzzles (7 rating bands). Solved = every move matched
         (any checkmating move also counts).
heldout  bench/heldout.pkl: 1,000 Lichess positions, every legal move graded by SF 17.1 depth 12.
         wp_lost = win prob given up vs the best move, top1 = picked a best move.
match    vs Stockfish with UCI_LimitStrength at the given UCI_Elo levels (--movetime, 0.1 s/move), colours
         alternate on the same random opening. "random" plays a random mover.
vs       head-to-head against another model (--vs path, or v1:models/v1 for VibeFish v1).
"""
import argparse
import csv
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from teacher import STOCKFISH, to_grade

_pick = None
_opp = None


def make_pick(spec):
    """spec = ("v2", model_path, mix, strip) or ("v1", checkout_dir, None, False) -> pick(board) -> move."""
    kind, path, mix, strip = spec
    torch.set_num_threads(1)
    if kind == "v1":
        mspec = importlib.util.spec_from_file_location("v1model", os.path.join(path, "model.py"))
        v1 = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(v1)
        net = v1.ChessGPT()
        weights = os.path.join(path, "model.pt")   # models/v1; an old v1 checkout keeps it in run/
        net.load_state_dict(torch.load(weights if os.path.exists(weights) else os.path.join(path, "run/model.pt")))
        net.eval()
        return lambda b: net.pick(b)
    from model import load, without_history
    net = load(path)
    return lambda b: net.pick(without_history(b) if strip else b, mix=mix)


def spec_of(arg):
    """'v1:<dir>' (e.g. v1:models/v1) -> v1 spec, anything else -> a model path, optionally 'path@mix'."""
    if arg.startswith("v1:"):
        return ("v1", arg[3:], None, False)
    path, _, mix = arg.partition("@")
    return ("v2", path, float(mix or 0), False)


def _init(spec, opp=None):
    global _pick, _opp
    _pick = make_pick(spec)
    _opp = make_pick(opp) if opp else None


def _puzzle(row):
    board = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    for i, uci in enumerate(moves):
        if i % 2 == 1:
            mine = _pick(board)
            if mine.uci() != uci:
                board.push(mine)
                return board.is_checkmate()
        board.push_uci(uci)
    return True


def puzzles(pool, path="bench/puzzles.csv"):
    rows = list(csv.DictReader(open(path)))
    ok = pool.map(_puzzle, rows, chunksize=8)
    out = {"puzzles": round(float(np.mean(ok)), 4)}
    band = lambda r: min(max(int(r["Rating"]), 400), 2800) // 400 * 400   # same bands as data.py puzzles
    for lo in range(400, 3200, 400):
        out[f"puz_{lo}"] = round(float(np.mean([o for o, r in zip(ok, rows) if band(r) == lo])), 3)
    return out


def _held(rec):
    fen, moves, scores = rec
    board = chess.Board(fen)
    for m in moves:
        board.push_uci(m)
    mv = _pick(board)
    g = {u: float(to_grade(s).clip(0, 1)) for u, s in scores.items()}
    best = max(g.values())
    chosen = g.get(mv.uci(), 0.0)
    return best - chosen, chosen >= best - 1e-9


def heldout(pool, path="bench/heldout.pkl"):
    res = pool.map(_held, pickle.load(open(path, "rb")), chunksize=16)
    return {"wp_lost": round(float(np.mean([r[0] for r in res])), 4),
            "top1": round(float(np.mean([r[1] for r in res])), 4)}


def _game(args):
    elo, g, movetime = args
    rng = random.Random(g // 2)
    board = chess.Board()
    for _ in range(4):   # same 4 random opening plies for both games of a pair
        board.push(rng.choice(list(board.legal_moves)))
    bot_white = g % 2 == 0
    engine = None
    if elo not in ("random", "vs"):
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        engine.configure({"Threads": 1, "UCI_LimitStrength": True, "UCI_Elo": int(elo)})
    try:
        while not board.is_game_over(claim_draw=True) and board.ply() < 400:
            if (board.turn == chess.WHITE) == bot_white:
                move = _pick(board)
            elif engine:
                move = engine.play(board, chess.engine.Limit(time=movetime)).move
            elif elo == "vs":
                move = _opp(board)
            else:
                move = random.choice(list(board.legal_moves))
            board.push(move)
    finally:
        if engine:
            engine.quit()
    res = board.result(claim_draw=True)
    pts = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
    return (pts if bot_white else 1 - pts), board


def match(pool, elo, games, pgn=None, me="VibeFish", opp=None, movetime=0.1):
    res = pool.map(_game, [(elo, g, movetime) for g in range(games)], chunksize=1)
    pts = [r[0] for r in res]
    s = float(np.mean(pts))
    opp = opp or f"opp {elo}"
    out = {"opp": opp if elo == "vs" else elo, "movetime": movetime, "W": pts.count(1.0), "D": pts.count(0.5), "L": pts.count(0.0), "score": round(s, 3)}
    if elo not in ("random", "vs"):
        sc = min(max(s, 0.5 / games), 1 - 0.5 / games)   # avoid infinities on 0% / 100%
        out["perf_elo"] = round(int(elo) + 400 * math.log10(sc / (1 - sc)))
    if pgn:
        with open(pgn, "a") as f:
            for i, (_, b) in enumerate(res):
                game = chess.pgn.Game.from_board(b)
                game.headers["Result"] = b.result(claim_draw=True)   # from_board leaves claimable draws as "*"
                game.headers["White"], game.headers["Black"] = (me, opp) if i % 2 == 0 else (opp, me)
                print(game, file=f, end="\n\n")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/vibefish_v3.pt")
    p.add_argument("--v1", help="VibeFish v1 dir, e.g. models/v1 (evaluates v1 instead)")
    p.add_argument("--mix", type=float, default=0.0, help="weight of the action-value head in the move choice")
    p.add_argument("--strip-history", action="store_true", help="hide move history from the model")
    p.add_argument("--puzzles", action="store_true")
    p.add_argument("--heldout", action="store_true")
    p.add_argument("--match", default="", help="comma list of UCI_Elo levels and/or 'random'")
    p.add_argument("--vs", help="also play a head-to-head match vs this model (path or path@mix, or v1:models/v1)")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--movetime", type=float, default=0.1, help="Stockfish seconds per move")
    p.add_argument("--pgn")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--log", help="append results as JSON here")
    a = p.parse_args()
    spec = ("v1", a.v1, None, False) if a.v1 else ("v2", a.model, a.mix, a.strip_history)
    out = {"model": a.v1 or a.model, "mix": a.mix, "strip_history": a.strip_history}
    opps = [e for e in a.match.split(",") if e] + (["vs"] if a.vs else [])
    with mp.Pool(a.workers, _init, (spec, a.vs and spec_of(a.vs))) as pool:
        if a.heldout:
            out.update(heldout(pool)); print(out, flush=True)
        if a.puzzles:
            out.update(puzzles(pool)); print(out, flush=True)
        out["matches"] = [match(pool, e, a.games, a.pgn, a.v1 or a.model, a.vs if e == "vs" else None, a.movetime) for e in opps]
    print(json.dumps(out))
    if a.log:
        with open(a.log, "a") as f:
            f.write(json.dumps(out) + "\n")

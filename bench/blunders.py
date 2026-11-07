"""Where does a model lose win probability? Grades every move the model made in match PGNs with
Stockfish and splits the loss by game phase (PLAN v4 item E16).

  STOCKFISH=... python bench/blunders.py games/v4_main_ema_sf.pgn --depth 12
"""
import argparse
import multiprocessing as mp
import os
import sys

import chess
import chess.pgn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import teacher

PIECE = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def phase(board):
    """opening = first 10 moves each; endgame = at most 13 points of non-pawn material left (both sides)."""
    if board.ply() < 20:
        return "opening"
    mat = sum(PIECE.get(p.piece_type, 0) for p in board.piece_map().values())
    return "endgame" if mat <= 13 else "middlegame"


def _init(depth, skip):
    global _engine, _depth, _skip
    _engine, _depth, _skip = teacher.open_engine(), depth, skip


def _game(args):
    """[(phase, wp lost, ply)] for every move the model played in this game."""
    moves, colour = args
    board, out = chess.Board(), []
    for m in moves:
        if board.turn == colour and board.ply() >= _skip and board.legal_moves.count() > 1:
            g = {u: float(teacher.to_grade(s).clip(0, 1)) for u, s in teacher.grade(_engine, board, _depth).items()}
            out.append((phase(board), max(g.values()) - g.get(m.uci(), min(g.values())), board.ply()))
        board.push(m)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pgn", nargs="+")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--player", default="vibefish", help="substring of the model's name in the White/Black tags")
    p.add_argument("--skip", type=int, default=4, help="random opening plies evaluate.py plays first")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    jobs = []
    for path in a.pgn:
        with open(path) as f:
            while g := chess.pgn.read_game(f):
                colour = chess.WHITE if a.player in g.headers["White"] else chess.BLACK
                jobs.append((list(g.mainline_moves()), colour))
    with mp.Pool(a.workers, _init, (a.depth, a.skip)) as pool:
        rows = [r for game in pool.map(_game, jobs, chunksize=1) for r in game]
    print(f"{len(jobs)} games, {len(rows)} model moves, depth {a.depth}")
    print(f"{'phase':<11} {'moves':>6} {'avg lost':>9} {'share of loss':>14} {'>0.1':>6} {'>0.3':>6}")
    total = sum(l for _, l, _ in rows)
    for ph in ("opening", "middlegame", "endgame"):
        ls = np.array([l for f, l, _ in rows if f == ph])
        if len(ls):
            print(f"{ph:<11} {len(ls):>6} {ls.mean():>9.4f} {ls.sum() / total:>13.0%} {(ls > 0.1).sum():>6} {(ls > 0.3).sum():>6}")


if __name__ == "__main__":
    main()

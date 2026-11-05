"""Stockfish grades every legal move. Scores are stored raw (centipawns, MATE-encoded mates)
and turned into grades (win prob, mates outside [0, 1]) only at training time."""
import os
import random
import chess
import chess.engine
import numpy as np
import torch

STOCKFISH = os.environ.get("STOCKFISH", "stockfish")  # path to a Stockfish binary
MATE = 100_000   # mate in n for side to move -> MATE - n; getting mated in n -> -MATE + n


def score_int(s: chess.engine.Score) -> int:
    m = s.mate()
    if m is None:
        return s.score()
    return MATE - m if m > 0 else -MATE - m


def to_grade(s):
    """Raw score(s) -> grade(s). cp -> Lichess win-prob curve; mate in n -> 1 + 0.2/n (shorter is
    better); mated in n -> -0.2/n. Works on ints or numpy arrays."""
    s = np.asarray(s, dtype=np.float64)
    wp = 1 / (1 + np.exp(-0.00368208 * np.clip(s, -5000, 5000)))
    win = 1 + 0.2 / np.maximum(MATE - s, 1)
    loss = -0.2 / np.maximum(s + MATE, 1)
    return np.where(s >= MATE - 1000, win, np.where(s <= -MATE + 1000, loss, wp))


def open_engine():
    e = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    e.configure({"Threads": 1, "Hash": 16})
    return e


def grade(engine, board: chess.Board, depth: int) -> dict[str, int]:
    """Raw score (side to move's view) after each legal move, keyed by UCI."""
    n = board.legal_moves.count()
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=n)
    return {info["pv"][0].uci(): score_int(info["score"].relative) for info in infos if "pv" in info}


# --- worker process state ----------------------------------------------------
_engine = _model = None
_depth = 8


def init_worker(depth, model_path=None):
    global _engine, _model, _depth
    torch.set_num_threads(1)
    _depth = depth
    _engine = open_engine()   # exits by itself when the worker's pipe closes
    if model_path:
        from model import load
        _model = load(model_path)


def grade_fen(fen):
    board = chess.Board(fen)
    if board.is_game_over():
        return None
    return fen, [], grade(_engine, board, _depth)


HISTORY = 12   # plies of history kept with each game position (enough for repetitions / last moves)


def play_game(seed, max_plies=200):
    """The model plays itself (sampling early so games differ); Stockfish grades every position.
    Returns records (fen HISTORY plies back, moves since then, scores) and the avg wp the model lost."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    board = chess.Board()
    fens, records, lost = [board.fen()], [], []
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        scores = grade(_engine, board, _depth)
        start = max(0, board.ply() - HISTORY)
        records.append((fens[start], [m.uci() for m in board.move_stack[start:]], scores))
        temp = 1.0 if board.ply() < 10 else (0.3 if rng.random() < 0.25 else 0.0)
        move = _model.pick(board, temperature=temp)
        best = max(scores.values())
        if move.uci() in scores:
            lost.append(float(to_grade(best).clip(0, 1) - to_grade(scores[move.uci()]).clip(0, 1)))
        board.push(move)
        fens.append(board.fen())
    return records, lost

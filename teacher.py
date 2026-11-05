"""Stockfish grades every legal move; games are played by a mix of the model and the teacher."""
import atexit
import math
import os
import random
import chess
import chess.engine
import torch

from model import ChessGPT, encode, orient_move, unorient_index

STOCKFISH = os.environ.get("STOCKFISH", "stockfish")  # path to a Stockfish binary


def cp_to_wp(cp: float) -> float:
    """Centipawns -> win probability (Lichess's curve)."""
    return 1 / (1 + math.exp(-0.00368208 * cp))


def open_engine():
    e = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    e.configure({"Threads": 1, "Hash": 16})
    return e


def grade(engine, board: chess.Board, depth: int) -> dict[int, float]:
    """Win probability (for side to move) after each legal move, keyed by oriented move index."""
    n = board.legal_moves.count()
    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=n)
    out = {}
    for info in infos:
        if "pv" not in info:
            continue
        m = info["pv"][0]
        if m.promotion not in (None, chess.QUEEN):
            continue  # under-promotions are folded into the queen promotion
        out[orient_move(board, m)] = cp_to_wp(info["score"].relative.score(mate_score=10000))
    return out


# --- worker process state -------------------------------------------------
_engine = None
_model = None


def init_worker(model_path):
    global _engine, _model
    torch.set_num_threads(1)
    _engine = open_engine()
    atexit.register(_engine.quit)
    if model_path:
        _model = ChessGPT()
        _model.load_state_dict(torch.load(model_path))
        _model.eval()


def play_game(args):
    """Play one game. Each ply: Stockfish grades the position (that's the training label),
    then the move is the model's (prob 1-beta) or the teacher's (prob beta)."""
    beta, temperature, depth, max_plies, seed = args
    rng = random.Random(seed)
    board = chess.Board()
    samples, model_plies, wp_lost = [], 0, 0.0
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        g = grade(_engine, board, depth)
        if not g:
            break
        samples.append((board.fen(), g))
        best = max(g.values())
        if _model is None or rng.random() < beta:
            # teacher move; early on pick among near-best moves so games differ
            tol = 0.04 if board.ply() < 12 else 0.0
            cands = [i for i, w in g.items() if w >= best - tol]
            idx = rng.choice(cands)
            move = unorient_index(board, idx)
        else:
            move = _model.pick(board, temperature)
            idx = orient_move(board, move)
            model_plies += 1
            wp_lost += best - g.get(idx, best)
        board.push(move)
    return samples, model_plies, wp_lost

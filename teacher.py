"""Stockfish grades every legal move. Scores are stored raw (centipawns, MATE-encoded mates)
and turned into grades (win prob, mates outside [0, 1]) only at training time."""
import os
import random
import subprocess
import chess
import numpy as np
import torch

STOCKFISH = os.environ.get("STOCKFISH", "stockfish")  # path to a Stockfish binary
MATE = 100_000   # mate in n for side to move -> MATE - n; getting mated in n -> -MATE + n


def to_grade(s):
    """Raw score(s) -> grade(s). cp -> Lichess win-prob curve; mate in n -> 1 + 0.2/n (shorter is
    better); mated in n -> -0.2/n. Works on ints or numpy arrays."""
    s = np.asarray(s, dtype=np.float64)
    wp = 1 / (1 + np.exp(-0.00368208 * np.clip(s, -5000, 5000)))
    win = 1 + 0.2 / np.maximum(MATE - s, 1)
    loss = -0.2 / np.maximum(s + MATE, 1)
    return np.where(s >= MATE - 1000, win, np.where(s <= -MATE + 1000, loss, wp))


class Engine:
    """Bare-bones UCI pipe. python-chess's analyse() validates every PV move, which costs as much
    CPU as Stockfish itself here; we only need each line's first move and score."""
    def __init__(self):
        self.p = subprocess.Popen([STOCKFISH], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci"); self.wait("uciok")
        self.send("setoption name Threads value 1\nsetoption name Hash value 16\nisready"); self.wait("readyok")

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def wait(self, prefix):
        for line in self.p.stdout:
            if line.startswith(prefix):
                return line


def open_engine():
    return Engine()


def grade(engine: Engine, board: chess.Board, depth: int) -> dict[str, int]:
    """Raw score (side to move's view) after each legal move, keyed by UCI. Sends the move history
    so Stockfish sees repetitions."""
    n = board.legal_moves.count()
    moves = " ".join(m.uci() for m in board.move_stack)
    engine.send(f"setoption name MultiPV value {n}\nposition fen {board.root().fen()}"
                + (f" moves {moves}" if moves else "") + f"\ngo depth {depth}")
    lines = {}
    for line in engine.p.stdout:
        if line.startswith("bestmove"):
            break
        t = line.split()
        if "multipv" not in t or "pv" not in t or "bound" in line:
            continue
        k = t.index("score")
        s = int(t[k + 2])
        if t[k + 1] == "mate":
            s = MATE - s if s > 0 else -MATE - s
        lines[t[t.index("multipv") + 1]] = (t[t.index("pv") + 1], s)   # deepest line per multipv wins
    return dict(lines.values())


# --- worker process state ----------------------------------------------------
_engine = _model = None
_depth = 8
_strip = False


def init_worker(depth, model_path=None, strip_history=False):
    global _engine, _model, _depth, _strip
    torch.set_num_threads(1)
    _depth, _strip = depth, strip_history
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
        view = board
        if _strip:   # a model trained only on history-free positions plays better without it
            view = chess.Board(board.fen())
            view.halfmove_clock = 0
        move = _model.pick(view, temperature=temp, mix=1.0)
        best = max(scores.values())
        if move.uci() in scores:
            lost.append(float(to_grade(best).clip(0, 1) - to_grade(scores[move.uci()]).clip(0, 1)))
        board.push(move)
        fens.append(board.fen())
    return records, lost

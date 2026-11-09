"""DeepMind's searchless_chess models (9M / 136M / 270M) as a minimal UCI engine, used only as an
opponent. Runs in its own venv with JAX:

  cd <dir with ../checkpoints/9M> && PYTHONPATH=<parent of searchless_chess> python dm_uci.py 9M

Their engine as published: one forward pass per legal move (action-value), argmax win prob. It ignores
time limits. Every `go` answers with that move.
"""
import sys

import chess
from searchless_chess.src.engines import constants

engine = constants.ENGINE_BUILDERS[sys.argv[1]]()
board = chess.Board()
for line in sys.stdin:
    cmd = line.split()
    if not cmd:
        continue
    if cmd[0] == "uci":
        print(f"id name DeepMind searchless {sys.argv[1]}\nuciok", flush=True)
    elif cmd[0] == "isready":
        print("readyok", flush=True)
    elif cmd[0] == "position":
        i = cmd.index("moves") if "moves" in cmd else len(cmd)
        board = chess.Board() if cmd[1] == "startpos" else chess.Board(" ".join(cmd[2:i]))
        for m in cmd[i + 1:]:
            board.push_uci(m)
    elif cmd[0] == "go":
        print(f"bestmove {engine.play(board).uci()}", flush=True)
    elif cmd[0] == "quit":
        break

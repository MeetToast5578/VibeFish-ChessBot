"""Play VibeFish. One forward pass per move, no search.

  python play.py                 # you vs VibeFish in the terminal (you are white; --black to swap)
  python play.py --uci           # UCI engine on stdin/stdout, for chess GUIs / lichess-bot
"""
import argparse
import sys

import chess
import torch

from model import load


def human(pick, human_white):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        print(board.unicode(invert_color=True, borders=True, orientation=human_white), "\n")
        if board.turn == human_white:
            try:
                board.push_san(input("your move (SAN, e.g. Nf3): ").strip())
            except ValueError:
                print("illegal or unreadable move")
            except EOFError:
                return
        else:
            move = pick(board)
            print("VibeFish plays", board.san(move))
            board.push(move)
    print(board.unicode(invert_color=True, borders=True, orientation=human_white))
    print("result:", board.result(claim_draw=True))


def uci(pick, name):
    board = chess.Board()
    for line in sys.stdin:
        cmd = line.split()
        if not cmd:
            continue
        if cmd[0] == "uci":
            print(f"id name {name}\nid author VibeFish\nuciok", flush=True)
        elif cmd[0] == "isready":
            print("readyok", flush=True)
        elif cmd[0] == "ucinewgame":
            board = chess.Board()
        elif cmd[0] == "position":
            rest = line.split(" moves ")
            board = chess.Board() if cmd[1] == "startpos" else chess.Board(rest[0].split(" fen ", 1)[1].strip())
            for m in rest[1].split() if len(rest) > 1 else []:
                board.push_uci(m)
        elif cmd[0] == "go":
            print(f"bestmove {pick(board).uci()}", flush=True)
        elif cmd[0] == "quit":
            return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/vibefish_v3.pt")
    p.add_argument("--uci", action="store_true")
    p.add_argument("--mix", type=float, default=0.0, help="weight of the action-value head in the move choice (still one forward pass)")
    p.add_argument("--black", action="store_true", help="play black against the bot")
    a = p.parse_args()
    torch.set_num_threads(1)
    net = load(a.model)
    pick = lambda b: net.pick(b, mix=a.mix)
    uci(pick, "VibeFish v3") if a.uci else human(pick, not a.black)

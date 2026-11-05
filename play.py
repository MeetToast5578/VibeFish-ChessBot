"""Match the trained model (no search, one forward pass per move) against an opponent.

  python3 play.py --opponent random --games 20
  python3 play.py --opponent sf --skill 0 --games 20     # Stockfish at Skill Level 0..20
"""
import argparse
import math
import random
import chess
import chess.engine
import chess.pgn
import torch

from model import ChessGPT, encode, legal_indices, unorient_index


@torch.no_grad()
def pick_search(board, k=5):
    """Model's top-k moves -> take any mate in 1, skip repetitions, else best value-head score."""
    logits, _ = model(torch.tensor([encode(board)]))
    legal = legal_indices(board)
    top = [legal[i] for i in logits[0, legal].topk(min(k, len(legal))).indices.tolist()]
    moves = [unorient_index(board, i) for i in top]
    children, scored = [], []
    for m in moves:
        board.push(m)
        if board.is_checkmate():
            board.pop(); return m
        rep = board.is_repetition(2) or board.can_claim_draw()
        children.append(encode(board)); scored.append((m, rep))
        board.pop()
    _, v = model(torch.tensor(children))          # v = win prob for the opponent
    best = max(range(len(scored)), key=lambda i: (1 - v[i].item()) - (0.5 if scored[i][1] else 0))
    return scored[best][0]
from teacher import STOCKFISH

p = argparse.ArgumentParser()
p.add_argument("--model", default="run/model.pt")
p.add_argument("--opponent", choices=["random", "sf"], default="sf")
p.add_argument("--skill", type=int, default=0)
p.add_argument("--games", type=int, default=20)
p.add_argument("--search", action="store_true", help="1-ply lookahead with the value head")
p.add_argument("--pgn", default=None, help="save games here")
args = p.parse_args()

torch.set_num_threads(1)
model = ChessGPT()
model.load_state_dict(torch.load(args.model))
model.eval()

engine = None
if args.opponent == "sf":
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"Threads": 1, "Skill Level": args.skill})

score, results = 0.0, {"win": 0, "draw": 0, "loss": 0}
rng = random.Random(0)
pgn_out = open(args.pgn, "w") if args.pgn else None
for g in range(args.games):
    board = chess.Board()
    model_white = g % 2 == 0
    # 2 random opening plies so the games differ
    for _ in range(2):
        board.push(rng.choice(list(board.legal_moves)))
    while not board.is_game_over(claim_draw=True) and board.ply() < 300:
        if (board.turn == chess.WHITE) == model_white:
            move = pick_search(board) if args.search else model.pick(board)
        elif engine:
            move = engine.play(board, chess.engine.Limit(time=0.05)).move
        else:
            move = rng.choice(list(board.legal_moves))
        board.push(move)
    res = board.result(claim_draw=True)
    if res == "*":
        res = "1/2-1/2"
    pts = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
    pts = pts if model_white else 1 - pts
    score += pts
    results[{1.0: "win", 0.5: "draw", 0.0: "loss"}[pts]] += 1
    print(f"game {g + 1}: model {'white' if model_white else 'black'}  {res}", flush=True)
    if pgn_out:
        game = chess.pgn.Game.from_board(board)
        game.headers["White"] = "ChessGPT" if model_white else f"opponent:{args.opponent}"
        game.headers["Black"] = f"opponent:{args.opponent}" if model_white else "ChessGPT"
        print(game, file=pgn_out, end="\n\n")

s = score / args.games
elo = 400 * math.log10(s / (1 - s)) if 0 < s < 1 else float("inf") * (1 if s else -1)
print(f"\n{results}  score {s:.0%}  -> Elo difference vs opponent ≈ {elo:+.0f}")
if engine:
    engine.quit()

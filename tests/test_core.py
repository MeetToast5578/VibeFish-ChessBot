import json
import random

import chess
import numpy as np
import torch

import data
import teacher
from model import ChessGPT, encode, legal_indices, orient_move, unorient_index


def random_boards(n=30, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        b = chess.Board()
        for _ in range(rng.randint(0, 60)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        yield b


def test_move_index_round_trip():
    for b in random_boards():
        for m in b.legal_moves:
            back = unorient_index(b, orient_move(b, m))
            assert back.from_square == m.from_square and back.to_square == m.to_square


def test_encoding_is_colour_symmetric():
    # mirror() swaps colours and side to move: the side-to-move view must be identical
    for b in random_boards():
        plain = chess.Board(b.fen())   # drop history: mirror() has none
        assert (encode(plain) == encode(plain.mirror())).all()


def test_mates_rank_by_distance():
    M = teacher.MATE
    g = teacher.to_grade([M - 1, M - 2, M - 10, 3000, 0, -3000, -M + 10, -M + 1])
    assert (np.diff(g) < 0).all()
    assert g[3] < 1 and g[2] > 1


def test_lichess_scores_flip_to_side_to_move():
    line = json.dumps({"fen": "4k3/8/8/8/8/8/4q3/K7 b - -",
                       "evals": [{"pvs": [{"cp": -500, "line": "e2e1"}, {"mate": -1, "line": "e2b2"}], "depth": 30}]})
    fen, moves, scores = data.parse_lichess(line)
    assert scores == {"e2e1": 500, "e2b2": teacher.MATE - 1}


def test_repetition_and_last_move_features():
    b = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        b.push_uci(uci)
    x = encode(b)
    assert x[67, 0] == 1                       # start position seen once before
    assert x[chess.F6, 6] == 1 and x[chess.G8, 6] == 2   # opponent's last move f6g8 (white to move: no mirroring)
    assert x[chess.F3, 6] == 3 and x[chess.G1, 6] == 4   # my previous move f3g1


def test_model_picks_legal_moves():
    net = ChessGPT(d=32, layers=1, heads=2).eval()
    for b in random_boards(5):
        if not b.is_game_over():
            assert net.pick(b) in b.legal_moves
            assert net.pick(b, mix=1.0) in b.legal_moves
    p, a, v = net(torch.from_numpy(encode(chess.Board()))[None])
    assert p.shape == a.shape == (1, 4096) and v.shape == (1,)


def test_head_to_head_game_uses_both_players():
    import evaluate
    seen = {"me": 0, "opp": 0}
    def player(who):
        def pick(b):
            seen[who] += 1
            return random.choice(list(b.legal_moves))
        return pick
    evaluate._pick, evaluate._opp = player("me"), player("opp")
    pts, board = evaluate._game(("vs", 1))
    assert pts in (0.0, 0.5, 1.0)
    assert board.is_game_over(claim_draw=True) or board.ply() >= 400
    assert seen["me"] > 0 and seen["opp"] > 0

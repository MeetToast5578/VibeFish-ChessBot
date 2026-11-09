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


def test_block_matches_torch_layer_and_grow_is_identity():
    import model
    torch.manual_seed(0)
    net = ChessGPT(d=32, layers=2, heads=4).eval()
    h = torch.randn(3, model.SEQ, 32)
    assert torch.allclose(net.body(h), model.block(net.body.layers[1], model.block(net.body.layers[0], h)), atol=1e-5)
    x = torch.from_numpy(np.stack([encode(b) for b in random_boards(4)]))
    big = model.grow(net, 3, bias=True).eval()
    assert big.cfg["layers"] == 3 and big.attn_bias.shape == (3, 4, model.SEQ, model.SEQ)
    for a, b in zip(net(x), big(x)):
        assert torch.allclose(a, b, atol=1e-5)
    looped = model.grow(net, 3, loops=2, bias=True)
    looped.train()
    sum(t.sum() for t in looped(x)).backward()   # bias and loop offsets get gradients
    assert looped.attn_bias.grad.abs().sum() > 0 and looped.loop_emb.grad.abs().sum() > 0


def test_head_to_head_game_uses_both_players():
    import evaluate
    seen = {"me": 0, "opp": 0}
    def player(who):
        def pick(b):
            seen[who] += 1
            return random.choice(list(b.legal_moves))
        return pick
    evaluate._pick, evaluate._opp = player("me"), player("opp")
    pts, board = evaluate._game(("vs", 1, 0.1))
    assert pts in (0.0, 0.5, 1.0)
    assert board.is_game_over(claim_draw=True) or board.ply() >= 400
    assert seen["me"] > 0 and seen["opp"] > 0


def test_chessbench_bag_reader_and_best_move_records(tmp_path):
    fens = ["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", chess.STARTING_FEN]
    recs = [bytes([len(f)]) + f.encode() + m.encode() for f, m in zip(fens, ["e7e5", "g1f3"])]
    ends = np.cumsum([len(r) for r in recs])
    path = tmp_path / "x.bag"
    path.write_bytes(b"".join(recs) + ends.astype("<i8").tobytes())
    got = [data.parse_bc(r) for r in data.bag_records(str(path), 0, 10)]
    assert got == [(f, [], {m: teacher.UNKNOWN}) for f, m in zip(fens, ["e7e5", "g1f3"])]
    assert [data.parse_bc(r)[0] for r in data.bag_records(str(path), 1, 1)] == fens[1:]
    assert np.isnan(teacher.to_grade(teacher.UNKNOWN)) and teacher.to_grade(0) == 0.5


def test_opponent_spec_takes_a_mix():
    import evaluate
    assert evaluate.spec_of("models/a.pt@2") == ("v2", "models/a.pt", 2.0, False)
    assert evaluate.spec_of("models/a.pt") == ("v2", "models/a.pt", 0.0, False)
    assert evaluate.spec_of("v1:models/v1")[:2] == ("v1", "models/v1")
    assert evaluate.spec_of("uci:sh dm.sh 9M")[:2] == ("uci", "sh dm.sh 9M")


def test_candidate_mode_adds_without_changing_one_pass_and_scores_legal_moves():
    from model import with_cand
    torch.manual_seed(0)
    m = ChessGPT(d=32, layers=2, heads=2, bias=True).eval()
    c = with_cand(m).eval()
    for b in random_boards(8, seed=3):
        if b.is_game_over():
            continue
        x = torch.from_numpy(encode(b))[None]
        for a, z in zip(m(x), c(x)):   # the one-pass outputs are untouched
            assert torch.allclose(a, z)
        legal, s = c.scores(b, cand=True)
        assert legal == legal_indices(b) and s.shape == (len(legal),)
        one = torch.stack([c(x, torch.tensor([i]))[0] for i in legal])   # batched == one pass per move
        assert torch.allclose(s, one, atol=1e-5)
        assert c.pick(b, cand=True) in b.legal_moves


def test_candidate_marks_change_the_output():
    from model import with_cand
    torch.manual_seed(0)
    c = with_cand(ChessGPT(d=32, layers=2, heads=2))
    torch.nn.init.normal_(c.cand_emb.weight)
    b = chess.Board()
    legal, s = c.scores(b, cand=True)
    assert len(set(s.tolist())) > 1   # different moves get different scores


def test_opponent_spec_takes_cand():
    import evaluate
    assert evaluate.spec_of("models/a.pt@cand") == ("v2", "models/a.pt", "cand", False)

"""VibeFish v2: a small transformer that reads a board and scores every from->to move.

The board is always shown from the side-to-move's point of view (black positions are
mirrored), so the net only has to learn chess "as white". Each of the 64 square tokens is
a sum of feature embeddings (piece, attackers, pins, last moves); 4 global tokens carry
castling, en passant, the 50-move clock and the repetition count.
"""
import math
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODING_VERSION = 2
# per-square features: piece (0 empty, 1-6 mine, 7-12 theirs), my / their attacker count (0-3),
# my / their cheapest attacker (0 none, 1-6 pawn..king), pinned to own king, last-move mark
# (0 none, 1/2 opponent's last move from/to, 3/4 my previous move from/to)
SQ_FEATS = [13, 4, 4, 7, 7, 2, 5]
# global tokens: castling bits (mine K, mine Q, theirs K, theirs Q), en-passant file (8 = none),
# halfmove clock // 10, repetition count of the current position (0, 1, 2+)
GLOBAL_FEATS = [16, 9, 10, 3]
NF = len(SQ_FEATS)
SEQ = 64 + len(GLOBAL_FEATS)


def orient_move(board: chess.Board, move: chess.Move) -> int:
    """Move index (from*64+to) in the side-to-move frame. Promotions map to from-to."""
    f, t = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    return f * 64 + t


def unorient_index(board: chess.Board, idx: int) -> chess.Move:
    f, t = divmod(idx, 64)
    if board.turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    move = chess.Move(f, t)
    p = board.piece_at(f)
    if p and p.piece_type == chess.PAWN and chess.square_rank(t) in (0, 7):
        move.promotion = chess.QUEEN
    return move


def legal_indices(board: chess.Board) -> list[int]:
    """Unique oriented move indices of legal moves (under-promotions collapse into queen promo)."""
    return sorted({orient_move(board, m) for m in board.legal_moves})


def encode(board: chess.Board) -> np.ndarray:
    """uint8 array [SEQ, NF]. Uses the move stack (if any) for last moves and repetitions."""
    us, them = board.turn, not board.turn
    flip = 0 if us == chess.WHITE else 56
    x = np.zeros((SEQ, NF), np.uint8)
    masks = {c: [board.pieces_mask(pt, c) for pt in chess.PIECE_TYPES] for c in (us, them)}
    for sq in chess.SQUARES:
        row = x[sq ^ flip]
        p = board.piece_at(sq)
        if p:
            row[0] = p.piece_type + (0 if p.color == us else 6)
            row[5] = board.is_pinned(p.color, sq)
        for col, c in ((1, us), (2, them)):
            a = board.attackers_mask(c, sq)
            if a:
                row[col] = min(a.bit_count(), 3)
                row[col + 2] = next(i + 1 for i, m in enumerate(masks[c]) if a & m)
    stack = board.move_stack
    if len(stack) >= 2:
        x[stack[-2].from_square ^ flip, 6], x[stack[-2].to_square ^ flip, 6] = 3, 4
    if stack:
        x[stack[-1].from_square ^ flip, 6], x[stack[-1].to_square ^ flip, 6] = 1, 2
    x[64, 0] = (board.has_kingside_castling_rights(us) + 2 * board.has_queenside_castling_rights(us)
                + 4 * board.has_kingside_castling_rights(them) + 8 * board.has_queenside_castling_rights(them))
    x[65, 0] = chess.square_file(board.ep_square) if board.has_legal_en_passant() else 8
    x[66, 0] = min(board.halfmove_clock, 99) // 10
    x[67, 0] = 2 if board.is_repetition(3) else 1 if board.is_repetition(2) else 0
    return x


class ChessGPT(nn.Module):
    def __init__(self, d=128, layers=4, heads=4):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads)
        self.sq_emb = nn.ModuleList(nn.Embedding(n, d) for n in SQ_FEATS)
        self.gl_emb = nn.ModuleList(nn.Embedding(n, d) for n in GLOBAL_FEATS)
        self.pos = nn.Parameter(torch.randn(SEQ, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.pq, self.pk = nn.Linear(d, d), nn.Linear(d, d)   # policy: from-square . to-square
        self.aq, self.ak = nn.Linear(d, d), nn.Linear(d, d)   # action value: same shape, sigmoid -> win prob
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.d = d

    def forward(self, x):
        """x: [B, SEQ, NF] ints -> policy logits [B, 4096], action-value logits [B, 4096], value [B]."""
        x = x.long()
        sq = sum(e(x[:, :64, i]) for i, e in enumerate(self.sq_emb))
        gl = torch.stack([e(x[:, 64 + j, 0]) for j, e in enumerate(self.gl_emb)], 1)
        h = self.norm(self.body(torch.cat([sq, gl], 1) + self.pos))
        s = h[:, :64]
        scale = 1 / math.sqrt(self.d)
        policy = (self.pq(s) @ self.pk(s).transpose(1, 2) * scale).flatten(1)
        action = (self.aq(s) @ self.ak(s).transpose(1, 2) * scale).flatten(1)
        value = torch.sigmoid(self.value(h.mean(1))).squeeze(-1)    # win prob for side to move
        return policy, action, value

    @torch.no_grad()
    def scores(self, board: chess.Board, mix=0.0):
        """(legal move indices, score per legal move). mix blends in the action-value head."""
        policy, action, _ = self(torch.from_numpy(encode(board))[None])
        legal = legal_indices(board)
        s = F.log_softmax(policy[0, legal], -1)
        if mix:
            s = s + mix * F.logsigmoid(action[0, legal])
        return legal, s

    @torch.no_grad()
    def pick(self, board: chess.Board, temperature=0.0, mix=0.0) -> chess.Move:
        legal, s = self.scores(board, mix)
        if temperature <= 0:
            i = legal[int(s.argmax())]
        else:
            i = legal[int(torch.multinomial(F.softmax(s / temperature, -1), 1))]
        return unorient_index(board, i)


def save(model: ChessGPT, path: str, **extra):
    sd = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"cfg": model.cfg, "encoding": ENCODING_VERSION, "state": sd, **extra}, path)


def load(path: str) -> ChessGPT:
    ck = torch.load(path, map_location="cpu")
    assert ck["encoding"] == ENCODING_VERSION, f"{path} uses encoding v{ck['encoding']}"
    model = ChessGPT(**ck["cfg"])
    model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    return model.eval()

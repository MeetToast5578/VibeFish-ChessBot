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


def without_history(board: chess.Board) -> chess.Board:
    """Same position with no move stack and a zero 50-move clock (what history-free training data looks like)."""
    b = chess.Board(board.fen())
    b.halfmove_clock = 0
    return b


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


def block(layer: nn.TransformerEncoderLayer, h, bias=None):
    """layer's pre-norm forward (dropout 0, gelu) with an optional [heads, S, S] additive attention bias."""
    a = layer.self_attn
    B, S, D = h.shape
    q, k, v = F.linear(layer.norm1(h), a.in_proj_weight, a.in_proj_bias).view(B, S, 3, a.num_heads, -1).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None if bias is None else bias.to(q.dtype))
    h = h + a.out_proj(o.transpose(1, 2).reshape(B, S, D))
    return h + layer.linear2(F.gelu(layer.linear1(layer.norm2(h))))


class ChessGPT(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, loops=1, bias=False):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, loops=loops, bias=bias)
        self.sq_emb = nn.ModuleList(nn.Embedding(n, d) for n in SQ_FEATS)
        self.gl_emb = nn.ModuleList(nn.Embedding(n, d) for n in GLOBAL_FEATS)
        self.pos = nn.Parameter(torch.randn(SEQ, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        # loops: run the layer stack this many times (shared weights, learned offset per pass)
        self.loop_emb = nn.Parameter(torch.zeros(loops, d)) if loops > 1 else None
        # learned additive attention bias per layer, head and square pair (a light "smolgen")
        self.attn_bias = nn.Parameter(torch.zeros(layers, heads, SEQ, SEQ)) if bias else None
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
        h = torch.cat([sq, gl], 1) + self.pos
        for p in range(self.cfg["loops"]):
            if self.loop_emb is not None:
                h = h + self.loop_emb[p]
            for i, layer in enumerate(self.body.layers):
                h = block(layer, h, None if self.attn_bias is None else self.attn_bias[i])
        h = self.norm(h)
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


def grow(src: ChessGPT, layers, loops=1, bias=False) -> ChessGPT:
    """Depth up-scaling: src's weights in a deeper model. Added layers start as identity
    (out_proj and linear2 zeroed, pre-norm), bias and loop offsets start at zero."""
    m = ChessGPT(src.cfg["d"], layers, src.cfg["heads"], loops, bias)
    missing, unexpected = m.load_state_dict(src.state_dict(), strict=False)
    assert not unexpected, unexpected
    for layer in m.body.layers[src.cfg["layers"]:]:
        for lin in (layer.self_attn.out_proj, layer.linear2):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
    return m


def load(path: str) -> ChessGPT:
    ck = torch.load(path, map_location="cpu")
    assert ck["encoding"] == ENCODING_VERSION, f"{path} uses encoding v{ck['encoding']}"
    model = ChessGPT(**ck["cfg"])
    model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    return model.eval()

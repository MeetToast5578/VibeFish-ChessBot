"""Small transformer that reads a board and scores every from->to move.

Board is always shown from the side-to-move's point of view (black positions are
mirrored), so the net only has to learn chess "as white".
"""
import math
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

N_PIECE = 13 * 16     # piece (0 empty, 1-6 mine, 7-12 theirs) x my attackers (0-3) x their attackers (0-3)
N_CASTLE = 16         # 4 castling-right bits
N_EP = 9              # en-passant file 0-7, 8 = none
SEQ = 66              # 64 squares + castling token + ep token


def orient(board: chess.Board) -> chess.Board:
    """Return board with side to move as white."""
    return board if board.turn == chess.WHITE else board.mirror()


def orient_move(board: chess.Board, move: chess.Move) -> int:
    """Move index (from*64+to) in the oriented frame. Promotions map to from-to."""
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


def encode(board: chess.Board) -> list[int]:
    b = orient(board)
    toks = [0] * 64
    for sq, p in b.piece_map().items():
        toks[sq] = p.piece_type + (0 if p.color == chess.WHITE else 6)
    for sq in range(64):
        mine = min(len(b.attackers(chess.WHITE, sq)), 3)
        theirs = min(len(b.attackers(chess.BLACK, sq)), 3)
        toks[sq] = toks[sq] * 16 + mine * 4 + theirs
    c = (b.has_kingside_castling_rights(chess.WHITE) * 1 + b.has_queenside_castling_rights(chess.WHITE) * 2
         + b.has_kingside_castling_rights(chess.BLACK) * 4 + b.has_queenside_castling_rights(chess.BLACK) * 8)
    ep = chess.square_file(b.ep_square) if b.ep_square is not None else 8
    return toks + [N_PIECE + c, N_PIECE + N_CASTLE + ep]


def legal_indices(board: chess.Board) -> list[int]:
    """Unique oriented move indices of legal moves (under-promotions collapse into queen promo)."""
    return sorted({orient_move(board, m) for m in board.legal_moves})


class ChessGPT(nn.Module):
    def __init__(self, d=128, layers=4, heads=4):
        super().__init__()
        self.tok = nn.Embedding(N_PIECE + N_CASTLE + N_EP, d)
        self.pos = nn.Parameter(torch.randn(SEQ, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)    # "move from this square" vectors
        self.k = nn.Linear(d, d)    # "move to this square" vectors
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.d = d

    def forward(self, x):
        h = self.norm(self.body(self.tok(x) + self.pos))
        sq = h[:, :64]
        logits = self.q(sq) @ self.k(sq).transpose(1, 2) / math.sqrt(self.d)   # B x 64 x 64
        v = torch.sigmoid(self.value(h.mean(1))).squeeze(-1)                    # win prob for side to move
        return logits.flatten(1), v

    @torch.no_grad()
    def pick(self, board: chess.Board, temperature=0.0) -> chess.Move:
        logits, _ = self(torch.tensor([encode(board)]))
        legal = legal_indices(board)
        l = logits[0, legal]
        if temperature <= 0:
            i = legal[int(l.argmax())]
        else:
            i = legal[int(torch.multinomial(F.softmax(l / temperature, -1), 1))]
        return unorient_index(board, i)

# VibeFish v1

A chess bot taught by Stockfish.

A small transformer learns chess from Stockfish's grades. In every position, Stockfish scores
**every legal move** (as a win probability). The model is trained to match that whole
distribution, not just copy the single best move.

## How the loop works (DAgger-style distillation)

```
round 0:  Stockfish plays itself       -> Stockfish grades every position -> train
round r:  the MODEL plays (Stockfish    -> Stockfish grades every position -> train on all
          steps in with prob beta)         the model reached               data so far
```

Grading the positions the model reaches is the key idea. A model trained only on Stockfish's
own games never sees the bad positions its own mistakes lead to, so it can't learn to
recover from them. Here Stockfish shows it the best way out of positions the model created.

This is denser than reinforcement learning. RL would give one "good/bad" number per move;
this gives a full grade for all ~30 legal moves at every position, at the same Stockfish cost.

## Files

| file | what it does |
|---|---|
| `model.py` | 0.88M-param transformer. 64 square tokens + castling + en-passant. The board is always shown from the side to move's view. Each square token also carries how many of my / their pieces attack it. Move head: `score(from,to) = q_from · k_to`. Value head: win probability. |
| `teacher.py` | Stockfish MultiPV grading, plus game play (model or teacher moves) in worker processes |
| `loop.py` | The rounds above. Logs to `run/log.jsonl`, saves `run/model.pt` and `run/data_round*.pkl` (FEN + grades, so data can be reused if the encoding changes) |
| `run/model.pt`, `run/log.jsonl` | VibeFish v1 weights and its training log |
| `games/games_vs_sf0.pgn` | 20 games vs Stockfish Skill 0 (policy only, no lookahead) |
| `play.py` | Matches vs a random mover or Stockfish `Skill Level` 0–20. `--search` adds a 1-move lookahead using the value head |

## Setup

```bash
pip install -r requirements.txt
git clone --depth 1 --branch sf_11 https://github.com/official-stockfish/Stockfish.git
cd Stockfish/src && make -j build ARCH=x86-64-modern && cd ../..
export STOCKFISH=$PWD/Stockfish/src/stockfish
python3 loop.py --rounds 5 --games 120 --epochs 2
python3 play.py --opponent sf --skill 0 --games 20 --search
```

(Stockfish 11 was used because it needs no network file download. Any modern Stockfish works the same way.)

## Results from this run (2 CPU cores, no GPU, about 1 hour)

Held-out set: 2,000 positions from Stockfish games, never trained on.
"wp lost" = how much win probability the model's chosen move gives up compared with Stockfish's best move.

| round | beta | positions | top-1 = Stockfish's move | wp lost / move | value error |
|---|---|---|---|---|---|
| untrained | – | 0 | 9% | 0.178 | 0.32 |
| 0 | 1.0 | 17k | 20.6% | 0.079 | 0.16 |
| 2 | 0.4 | 47k | 22.8% | 0.075 | 0.12 |
| 4 | 0.2 | 80k | 24.8% | 0.072 | 0.10 |

Matches (20 games each, 2 random opening plies):

| model mode | vs random mover | vs Stockfish 11 Skill 0 |
|---|---|---|
| policy only (1 forward pass/move) | +2 =18 −0 (draws by repetition) | +0 =3 −17 |
| `--search` (top-5 moves + value head) | **+13 =7 −0** | **+2 =2 −16** |

In short, it's a beginner. It knows how pieces move, sees hanging pieces, and wins against
random play, but it still loses most games to Stockfish's weakest setting.

### What mattered most for efficiency
* **Attack-count features.** On raw piece positions the model was barely better than picking a random legal move
  after 3 rounds (eval loss 2.89 vs 2.99 for uniform). With per-square attacker counts, 150 training steps
  beat the old 3-round result. Cheap hand-made features help enormously when compute is tiny.
* **Grading all moves (soft targets)** instead of only the best move.
* **Tiny lookahead** at play time: +35 → +269 Elo vs random for about 5 extra forward passes per move.

## How to make it much stronger
1. **Compute.** This run did about 1,500 gradient steps. Strong searchless models train for millions.
   On a single GPU, use `--games 2000 --epochs 4` and a bigger model (`d=256, layers=8`).
2. **Deeper labels.** Depth 8 → 12+ gives better grades (it costs more Stockfish time; parallelise with `--workers`).
3. **Lichess evaluated positions.** The Lichess database ships millions of positions with
   Stockfish evals, which can serve as a free pre-training set before the DAgger rounds.
4. **Repetition awareness.** Add the last few moves or a repetition-count token so it stops shuffling.
5. **Real search.** Use the value head inside a small alpha-beta or MCTS. This is where engines get most of their strength.

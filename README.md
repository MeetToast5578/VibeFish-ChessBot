# VibeFish v2

A small chess bot with **no search**. A 2.9M-parameter transformer looks at the board once and
picks a move. No alpha-beta, no MCTS, no lookahead. Every bit of strength is in the network.

```bash
python play.py              # play against it in the terminal
python play.py --uci        # UCI engine for chess GUIs / lichess-bot
```

## Results

All numbers are **policy only**: one forward pass per move. Benchmarks live in `bench/` and are
never trained on.

RESULTS_TABLE

* **Puzzles**: 1,050 Lichess puzzles, 150 in each rating band. Solved = every move of the solution
  found (any mate in 1 also counts).
* **Held-out**: 1,000 Lichess positions, every legal move graded by Stockfish 17.1 at depth 12.
  *wp lost* = win probability given up per move compared with Stockfish's best move.
* **Matches**: against Stockfish 17.1 with `UCI_LimitStrength` (0.1 s per move, so it plays below
  its nominal rating), colours alternating on the same random 4-ply openings.

## What changed from v1, and why

v1 lost 13 of 20 to the weakest Stockfish setting and drew 18 of 20 against a *random mover*.
The diagnosis (details in `PLAN.md`):

1. **It couldn't see repetitions.** Stockfish graded repeating moves as draws, but the board
   encoding had no history, so those labels looked like noise. v2 adds a repetition-count token,
   the 50-move clock, and markers for the last two moves.
2. **Every mate looked the same.** All mates were squashed to win probability 1.0, so mate in 1
   and mate in 15 were equally good and the bot never finished games. v2 grades mate in *n* as
   `1 + 0.2/n` (and getting mated as `-0.2/n`), so shorter mates clearly win.
3. **Far too little data.** v1 saw 80k positions graded at depth 8. v2 trains on 6M positions from
   the Lichess evaluation database (Stockfish at depth 20–50, top 1–5 moves each), plus ~220k
   positions with *every* legal move graded by Stockfish 17.1, plus its own games (DAgger).
4. **Richer cheap features.** Besides how many pieces attack a square, each square now knows the
   *cheapest* attacker on each side and whether a piece is pinned. Small nets need this help.
5. **Better training.** Warmup + cosine schedule, a separate action-value head (win prob for
   every move) next to the policy, and a value head.

## How it was trained (4 CPU cores, no GPU)

| stage | data | time |
|---|---|---|
| main | 6M Lichess-eval positions (60%), 100k Lichess positions + 119k puzzle positions with every move graded by SF 17.1 depth 8 (40%) | 250 min |
| DAgger | the main model plays itself, Stockfish grades every move it faced (with full history) | DAGGER_GEN |
| fine-tune | main mix + DAgger games | FT_TIME |

Measured budget on this box: ~110 training positions/s for the 2.9M model (fp32, bf16 is slower
without AMX), ~34 positions/s/core for Stockfish grading all moves at depth 8.

## Files

| file | what it does |
|---|---|
| `model.py` | board encoding (side-to-move view, 64 square tokens + 4 global tokens), the transformer, `pick()` |
| `teacher.py` | Stockfish grading of every legal move (lean UCI pipe), score → grade conversion, self-play worker |
| `data.py` | Lichess eval DB / puzzle positions / DAgger games → encoded `.npz` shards; builds the puzzle benchmark |
| `train.py` | trainer with a wall-clock budget, mixes several data sources |
| `evaluate.py` | puzzles, held-out positions, matches vs Stockfish `UCI_Elo` (also runs VibeFish v1 via `--v1`) |
| `play.py` | terminal play and UCI mode |
| `models/vibefish_v2.pt` | the weights (fp16, ~6 MB) |
| `bench/` | benchmark sets and `results.jsonl` with every evaluation run |
| `PLAN.md`, `CLAUDE.md` | roadmap / measurements, and notes for agents working on the repo |

## Reproduce

```bash
python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
# any modern Stockfish; v2 used the 17.1 release binary
export STOCKFISH=/path/to/stockfish
mkdir -p data
curl -r 0-1199999999 -o data/eval_part.zst https://database.lichess.org/lichess_db_eval.jsonl.zst
curl -o data/puzzles.csv.zst https://database.lichess.org/lichess_db_puzzle.csv.zst
python data.py lichess --start 20000 --count 6000000 --exclude bench/exclude.txt --out data/L
python data.py sf --start 20000 --count 100000 --out data/S_0
python data.py sf --src data/puzzles.csv.zst --start 0 --count 120000 --exclude bench/exclude.txt --out data/S_puz
python train.py --mix "data/L_*.npz:0.6,data/S_0.npz:0.15,data/S_puz.npz:0.25" --minutes 250 --d 192 --layers 6 --heads 6 --out runs/main
python data.py dagger --model runs/main/model.pt --games 2000 --out data/D1
python train.py --mix "data/L_*.npz:0.5,data/S_0.npz:0.1,data/S_puz.npz:0.2,data/D1.npz:0.2" --init runs/main/model.pt --lr 3e-4 --minutes 60 --out runs/ft
python evaluate.py --model runs/ft/model.pt --puzzles --heldout --match random,1320,1600
```

VibeFish v1 (the 0.88M model with a 1-ply lookahead option) is at commit `4d13eec`.

# VibeFish v3

A small chess bot with **no search**. A 6.7M-parameter transformer looks at the board once and
picks a move. No alpha-beta, no MCTS, no lookahead. Every bit of strength is in the network.

```bash
python play.py              # play against it in the terminal
python play.py --uci        # UCI engine for chess GUIs / lichess-bot
```

## Results

All numbers are **policy only**: one forward pass per move. Benchmarks live in `bench/` and are
never trained on.

| | VibeFish v1 | VibeFish v2 | **VibeFish v3** |
|---|---|---|---|
| parameters | 0.88M | 2.88M (5.8 MB file) | 6.69M (13.4 MB file) |
| puzzles solved (1,050) | 10.5% | 39.1% | **59.3%** |
| ↳ by rating 400 / 800 / 1200 / 1600 / 2000 / 2400 / 2800+ | 25 / 17 / 17 / 8 / 4 / 2 / 1 % | 95 / 77 / 58 / 28 / 9 / 5 / 3 % | **100 / 96 / 88 / 67 / 41 / 21 / 2 %** |
| held-out: win prob lost per move | 0.086 | 0.054 | **0.028** |
| held-out: picks Stockfish's best move | 33% | 42% | **52%** |
| vs random mover (120 games) | +19 =99 −2 | +106 =14 −0 | **+118 =2 −0** |
| vs Stockfish `UCI_Elo` 1320 (120 games) | +3 =23 −94 (12%) | +48 =29 −43 (52%) | **+105 =10 −5 (92%)** |
| vs Stockfish `UCI_Elo` 1600 (120 games) | – | +18 =26 −76 (26%) | **+79 =18 −23 (73%)** |
| vs Stockfish `UCI_Elo` 2000 (120 games) | – | +3 =22 −95 (12%) | **+45 =27 −48 (49%)** |

On Stockfish's limited-strength scale (at 0.1 s per move) v3 performs at **~1,740–1,990**
(1737 / 1776 / 1991 against SF 1320 / 1600 / 2000), v2 at ~1,350–1,650 and v1 at ~975. v3 scores
49% against SF 2000, where v2 scored 12%, with zero search.

**Round-robin** (100 games per pair, 50 random 4-ply openings played with both colours,
`games/roundrobin_v1_v2_v3.pgn`):

| | vs v1 | vs v2 | vs v3 |
|---|---|---|---|
| **v3** | +91 =9 −0 (95.5%) | +86 =12 −2 (92%) | – |
| v2 | +64 =36 −0 (82%) | – | +2 =12 −86 (8%) |
| v1 | +0 =36 −64 (18%) | +0 =9 −91 (4.5%) | – |

That's roughly +420 Elo for v3 over v2 and +260 for v2 over v1, head to head. 52 of the 57 draws
were threefold repetitions, 33 of them in v2 vs v1.
Sample games: `games/v3s2_matches.pgn` (v3 vs Stockfish), `games/v2_vs_sf1600.pgn`.

* **Puzzles**: 1,050 Lichess puzzles, 150 in each rating band. Solved = every move of the solution
  found (any mate in 1 also counts).
* **Held-out**: 1,000 Lichess positions, every legal move graded by Stockfish 17.1 at depth 12.
  *wp lost* = win probability given up per move compared with Stockfish's best move.
* **Matches**: against Stockfish 17.1 with `UCI_LimitStrength` (0.1 s per move, so it plays below
  its nominal rating), colours alternating on the same random 4-ply openings.

## What's new in v3

Same encoding and network design as v2, scaled up and trained on free Kaggle GPUs (2×T4, fp16):

1. **Bigger model:** 6.7M parameters (width 256, 8 layers, 8 heads) instead of 2.9M (192, 6, 6).
   Still well under the 10M cap.
2. **~30× more training:** ~67M positions seen (v2: ~2M), from 26M Lichess-eval positions
   (6M + 20M new), 1.6M puzzle positions with every legal move graded by Stockfish 17.1 at depth 8,
   and three rounds of self-play.
3. **Two stages:** a broad first stage (mostly Lichess evals), then a fine-tune weighted towards
   every-move-graded puzzle positions and the stage-1 model's own games (DAgger). The fine-tune
   lost a little on held-out positions (0.024 → 0.028 wp lost) but won more games at every level
   (SF 2000: 40% → 49%), so it shipped. Games decide.

## What changed from v1 to v2, and why

v1 drew 99 of 120 games against a *random mover* and scored 12% against Stockfish 1320.
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

## How v3 was trained (Kaggle, 2×T4)

| stage | data | time |
|---|---|---|
| stage 1 | Lichess evals L (30%) + L2 (50%), every-move-graded S_0 / S_puz (10%), v2 self-play D1 / D2 (10%); lr 1e-3, warmup 500, cosine, batch 1024 | 180 min, 39M positions |
| self-play D3 | the stage-1 model plays 3,000 games (half from puzzle-game positions); Stockfish 17.1 grades every move at depth 8 | ~218k positions, local CPU |
| stage 2 | from stage 1: S2 puzzle positions (35%), D3 (17%), L (15%), L2 (20%), S / D1 / D2 (13%); lr 3e-4, warmup 300, cosine | 120 min, 28M positions |

Training logs: `bench/train_logs/`. Every evaluation run: `bench/results.jsonl`.

## How v2 was trained (4 CPU cores, no GPU)

| stage | data | time |
|---|---|---|
| main | 6M Lichess-eval positions (60%), plus 100k Lichess positions and 119k puzzle positions with every move graded by SF 17.1 at depth 8 (40%) | 127 min (step 2000; the machine rebooted at ~2400) |
| DAgger | the step-2000 model plays 1,200 games against itself; Stockfish grades every position it reached, with full move history (97k positions) | 35 min |
| continuation | main mix (80%) + DAgger games (20%), lr 5e-4 → cosine, from the step-2000 weights | 150 min |

About 2M training positions in total. A second DAgger round plus a 30-minute polish was tried
and was **worse** in games (perf 1340 vs 1417 at SF 1600, 120 games each), so it wasn't shipped.
Training logs: `bench/train_logs/`.

**The bug that mattered most:** the Lichess and puzzle positions carry no move history, so the
history features (last moves, 50-move clock, repetitions) were never trained. In real games they
are always set, so the untrained embeddings scrambled the input. The checkpoint that solved
25% of puzzles lost **0–20** to SF 1320. Ten minutes of training on the DAgger games, which do
have history, fixed it (+6 =2 −12 on the same test, then 52% over 120 games after the full run).

Throughput on this box: ~130–220 training positions/s for the 2.9M model (fp32; bf16 is slower
without AMX) and ~34 positions/s/core for Stockfish grading every move at depth 8.

## Weaknesses / next steps

* Deep tactics are still weak: 41% of puzzles rated 2000, 21% at 2400, 2% at 2800+. v3 saw
  ~67M positions; DeepMind's searchless 9M model saw ~41 billion. Training volume is the main
  lever (`PLAN.md`, "Road to 3000"), starting with the ChessBench data in v3.1.
* The stage-2 fine-tune traded a little accuracy on ordinary game positions for tactics. The next
  fine-tune should keep more Lichess data in the mix and start from a lower learning rate.
* 2 of 120 games against a random mover were still draws.

## Files

| file | what it does |
|---|---|
| `model.py` | board encoding (side-to-move view, 64 square tokens + 4 global tokens), the transformer, `pick()` |
| `teacher.py` | Stockfish grading of every legal move (lean UCI pipe), score → grade conversion, self-play worker |
| `data.py` | Lichess eval DB / puzzle positions / DAgger games → encoded `.npz` shards; builds the puzzle benchmark |
| `train.py` | trainer with a wall-clock budget, mixes several data sources |
| `evaluate.py` | puzzles, held-out positions, matches vs Stockfish `UCI_Elo`, head-to-head vs another model (`--vs`; VibeFish v1 via `--v1 models/v1` / `--vs v1:models/v1`) |
| `kaggle_run.py` | runs commands on a Kaggle GPU notebook with datasets attached, downloads the outputs |
| `play.py` | terminal play and UCI mode |
| `models/vibefish_v3.pt` | the v3 weights (fp16, 13.4 MB) |
| `models/vibefish_v2.pt` | the v2 weights (fp16, ~6 MB) |
| `models/v1/` | VibeFish v1: its own `model.py` (different architecture) and weights (3.5 MB) |
| `bench/` | benchmark sets, `results.jsonl` with every evaluation run, training logs |
| `games/` | match games vs Stockfish and the v1 / v2 / v3 round-robin |
| `PLAN.md`, `CLAUDE.md` | roadmap / measurements, and notes for agents working on the repo |

## Reproduce

v3 was trained on Kaggle with `kaggle_run.py`. The exact commands, datasets and mixes are in
`PLAN.md` (v3 section) and the training logs. The CPU-only v2 recipe:

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
python train.py --mix "data/L_*.npz:0.6,data/S_0.npz:0.15,data/S_puz.npz:0.25" --minutes 130 --d 192 --layers 6 --heads 6 --out runs/main
python data.py dagger --model runs/main/model.pt --strip-history --games 1200 --out data/D1
python train.py --mix "data/L_*.npz:0.5,data/S_0.npz:0.1,data/S_puz.npz:0.2,data/D1.npz:0.2" --init runs/main/model.pt --lr 5e-4 --warmup 100 --minutes 150 --d 192 --layers 6 --heads 6 --out runs/cont
python evaluate.py --model runs/cont/model.pt --puzzles --heldout --match random,1320,1600 --games 120
```

VibeFish v1 (the 0.88M model with a 1-ply lookahead option) is at commit `4d13eec`.

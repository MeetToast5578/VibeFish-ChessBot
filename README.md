# VibeFish v4.1

A small chess bot with **no search**. A 13.6M-parameter transformer looks at the board once and
picks a move. No alpha-beta, no MCTS, no lookahead. Every bit of strength is in the network.

```bash
python play.py              # play against it in the terminal
python play.py --uci        # UCI engine for chess GUIs / lichess-bot
```

## Results

Every number is **one forward pass per move**. v1–v3 use the policy head only. v4 and v4.1 add the
action-value head's win-probability estimate for each move to the policy (`--mix 2`, the default).
Both heads come out of the same pass. Benchmarks live in `bench/` and are never trained on.

| | VibeFish v1 | VibeFish v2 | VibeFish v3 | **VibeFish v4** | **VibeFish v4.1** |
|---|---|---|---|---|---|
| parameters | 0.88M | 2.88M (5.8 MB file) | 6.69M (13.4 MB file) | **13.6M (27 MB file)** | **13.6M (27 MB file)** |
| puzzles solved (1,050) | 10.5% | 39.1% | 59.3% | **63.5%** | **64.3%** |
| ↳ by rating 400 / 800 / 1200 / 1600 / 2000 / 2400 / 2800+ | 25 / 17 / 17 / 8 / 4 / 2 / 1 % | 95 / 77 / 58 / 28 / 9 / 5 / 3 % | 100 / 96 / 88 / 67 / 41 / 21 / 2 % | **99 / 98 / 93 / 79 / 45 / 27 / 4 %** | **99 / 99 / 94 / 81 / 47 / 27 / 4 %** |
| held-out: win prob lost per move | 0.086 | 0.054 | 0.028 | **0.023** | **0.023** |
| held-out: picks Stockfish's best move | 33% | 42% | 52% | **55%** | **55%** |
| vs random mover (120 games) | +19 =99 −2 | +106 =14 −0 | +118 =2 −0 | **+117 =3 −0** | **+117 =3 −0** |
| vs Stockfish `UCI_Elo` 1320 (120 games) | +3 =23 −94 (12%) | +48 =29 −43 (52%) | +105 =10 −5 (92%) | **+113 =6 −1 (97%)** | +110 =8 −2 (95%) |
| vs Stockfish `UCI_Elo` 1600 (120 games) | – | +18 =26 −76 (26%) | +79 =18 −23 (73%) | **+86 =19 −15 (80%)** | **+91 =21 −8 (85%)** |
| vs Stockfish `UCI_Elo` 2000 (200 games) | – | +3 =22 −95 (12%, 120 games) | +72 =45 −83 (47%) | **+86 =53 −61 (56%)** | **+102 =52 −46 (64%)** |
| vs Stockfish `UCI_Elo` 2400 (200 games) | – | – | – | **+38 =56 −106 (33%)** | **+38 =56 −106 (33%)** |
| head-to-head vs v3 (200 games) | | | – | **+106 =54 −40 (66.5%)** | – |
| head-to-head vs v4 (200 games, both `--mix 2`) | | | | – | **+80 =67 −53 (56.8%)** |

**v4.1** is v4 fine-tuned for 150 min on a new self-play round, D5: 4,000 games by the v4 main-run model,
every move graded by Stockfish at **depth 12** instead of 8, 25% of the mix. It beats v4 56.8% head to head
(about +47 Elo) and scores 64% against SF 2000 (perf 2,100; v4 56%). Against SF 2400 it ties v4 (33%).

**v1.1** (`models/vibefish_v1_1_ema.pt`): v1's size (0.89M) with everything learned
since, trained from scratch for 3 h on one T4 (101M positions). Policy only:

| | old v1 | **v1.1** | v2 | v3 |
|---|---|---|---|---|
| parameters | 0.88M | **0.89M** | 2.9M | 6.7M |
| positions seen in training | ~0.4M | **101M** | ~2M | ~67M |
| compute | 2 CPU cores, 1 h | **1×T4, 3 h** | 4 CPU cores, ~4.5 h | 2×T4, 5 h |
| puzzles solved | 10.5% | **58.4%** | 39.1% | 59.3% |
| held-out wp lost / top1 | 0.086 / 33% | **0.031 / 50.5%** | 0.054 / 42% | 0.028 / 52% |
| vs random (120) | +19 =99 −2 | **+113 =7 −0** | +106 =14 −0 | +118 =2 −0 |
| vs SF 1320 / 1600 / 2000 (120) | 12% / – / – | **87.5% / 67.9% / 35.0%** | 52% / 26% / 12% | 92% / 73% / 49% |
| perf Elo vs SF 1320 / 1600 / 2000 | – | **1,658 / 1,730 / 1,892** (SF 2000 with `--mix 2`: 38.3%, 1,917) | – | – / – / 1,981 |
| head-to-head, 200 games | | **93.0% vs old v1, 89.0% vs v2, 38.5% vs v3** | | |

The same tiny network goes from drawing a random mover to beating v2 89%. **v1.1 plays at about
1,900** on Stockfish's `UCI_Elo` scale (0.1 s per move). The SF 2000 match is the one to trust, since
its score is closest to 50%; the lopsided wins against weaker levels understate it. v4.1 is ~2,100
there, at 15× the parameters. Details: `PAPER.md` §4.8.

On Stockfish's limited-strength scale (at 0.1 s per move) v4 performs at **~2,040** against
SF 2000 and 2,277 against SF 2400. v3 performs at 1,981 against SF 2000, v2 at ~1,350–1,650 and
v1 at ~975. Against weak opponents the perf numbers say little (v4 has 1 loss in 120 games
against SF 1320). Head to head, v4 beats v3 66.5%, about +120 Elo.

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

## What's new in v4

1. **Twice as deep:** 16 layers instead of 8 (width 256, 13.6M parameters; the owner raised the
   cap to 14M). It grew out of v3: the 8 new layers start as identity, so training begins from
   v3's play. A learned **square-pair attention bias** per layer and head is added.
2. **The new layers need a higher learning rate.** In a 2 h pilot they barely moved (weight norm
   ~3 vs ~70 for v3's layers). Training them at 5× lr (`train.py --boost-from 8 --boost 5`) fixed it.
3. **More and fresher data:** ~100M positions seen over the pilot, main run and fine-tune,
   with a third of each batch drawn from 60M fresh ChessBench positions (DeepMind, best move only),
   our own every-move-graded data, and a new self-play round (D4, 4,000 games, 311k positions).
4. **Weight averaging (EMA)** and saved optimizer state in `train.py`. The shipped weights are the EMA.
5. **Action-value blend at play time (`--mix 2`):** still one pass, but the move choice also uses
   the action-value head. +4–8% against Stockfish over policy only.

Where it still loses: in the middlegame. `bench/blunders.py` graded v4's moves against SF 2000.
The middlegame holds 70% of the lost win probability and 42 of its 50 moves that lose more
than 0.3; the endgame holds 18%. Details and every step: `PLAN.md` (v4 section).

## How v4 was trained (Kaggle, 2×T4)

| stage | data | time |
|---|---|---|
| pilot A | grown from v3.1-cb to 16 layers + attention bias; ChessBench CB2 (35%), Lichess L / L2, puzzle positions S2, self-play D1–D3 | 110 min, 14M positions |
| main | from pilot A, layers 8–15 at 5× lr; ChessBench CB3 (35%) + the same every-move-graded mix; lr 2e-4, cosine | 10.5 h, 63M positions |
| self-play D4 | pilot A plays 4,000 games (half from puzzle positions); SF 17.1 grades every move at depth 8 | 311k positions, Kaggle CPU |
| fine-tune | from the main run's EMA and optimizer state: ChessBench CB4 (30%), D4 (15%), unused Lichess L2_10–19, S2, D3; lr 1e-4 | 3 h, 23M positions |

## What was new in v3

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

* Middlegame tactics: 45% of puzzles rated 2000, 27% at 2400, 4% at 2800+, and 70% of the win
  probability v4 gives away against SF 2000 is in the middlegame. Our every-move labels come from
  Stockfish at depth 8. The next self-play round (D5) is graded at depth 12.
* v4's line has seen ~220M positions in all (v3, the v3.1 fine-tune and v4). DeepMind's searchless 9M model saw ~41
  billion. Training volume is still the main lever (`PLAN.md`).
* 3 of 120 games against a random mover were still draws.

## Files

| file | what it does |
|---|---|
| `model.py` | board encoding (side-to-move view, 64 square tokens + 4 global tokens), the transformer, `pick()` |
| `teacher.py` | Stockfish grading of every legal move (lean UCI pipe), score → grade conversion, self-play worker |
| `data.py` | Lichess eval DB / puzzle positions / DAgger games → encoded `.npz` shards; builds the puzzle benchmark |
| `train.py` | trainer with a wall-clock budget, mixes several data sources |
| `evaluate.py` | puzzles, held-out positions, matches vs Stockfish `UCI_Elo`, head-to-head vs another model (`--vs path` or `path@mix`; VibeFish v1 via `--v1 models/v1` / `--vs v1:models/v1`). Its `--mix` defaults to 0; v4's numbers use `--mix 2` |
| `bench/blunders.py` | win probability a model gave away in match games, by game phase |
| `kaggle_run.py` | runs commands on a Kaggle GPU notebook with datasets attached, downloads the outputs |
| `play.py` | terminal play and UCI mode |
| `models/vibefish_v4_1.pt` | **the shipped v4.1 weights** (fp16, 27 MB; copy of `vibefish_v41_ft_ema.pt`) |
| `models/vibefish_v4.pt` | the v4 weights (fp16, 27 MB; the fine-tune's EMA) |
| `models/vibefish_v41_ft.pt`, `vibefish_v41_ft_ema.pt` | the v4.1 fine-tune (plain weights / EMA) |
| `models/vibefish_v1_1.pt`, `vibefish_v1_1_ema.pt` | v1's size (0.89M) retrained with the current recipe |
| `models/vibefish_v3.pt` | the v3 weights (fp16, 13.4 MB) |
| `models/vibefish_v4_pilot_{a,b}.pt`, `vibefish_v4_main.pt`, `vibefish_v4_main_ema.pt`, `vibefish_v4_ft.pt`, `vibefish_v4_ft_ema.pt` | v4's steps, kept for comparison (`vibefish_v4.pt` is a copy of `_ft_ema`) |
| `models/vibefish_v2.pt` | the v2 weights (fp16, ~6 MB) |
| `models/vibefish_v3_stage1.pt`, `vibefish_v3_1_control.pt`, `vibefish_v3_1_cb.pt` | not shipped, kept for comparison: v3 stage 1, and the v3.1 fine-tunes without / with ChessBench (both even with v3) |
| `models/v1/` | VibeFish v1: its own `model.py` (different architecture) and weights (3.5 MB) |
| `bench/` | benchmark sets, `results.jsonl` with every evaluation run, training logs |
| `games/` | match games vs Stockfish and the v1 / v2 / v3 round-robin |
| `PAPER.md` | technical report: the whole process from v1 to v4.1, how the bot is built and how it works |
| `PLAN.md`, `CLAUDE.md` | roadmap / measurements, and notes for agents working on the repo |

## Credits

Trained in part on **ChessBench** (DeepMind, CC-BY 4.0): best-move records from the
`searchless_chess` dataset. DeepMind's models are not used, only their data.
Every-move grades come from Stockfish 17.1. Positions come from the Lichess evaluation and puzzle databases.

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

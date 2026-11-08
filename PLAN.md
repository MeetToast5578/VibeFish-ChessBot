# VibeFish plan

## Goal and rules

A **small, searchless** chess bot.

- **Searchless:** the network looks only at the *current* position (once, or a handful of
  times, e.g. a mirrored copy) and picks a move. Nothing evaluates positions after a candidate
  move: no 1-ply lookahead, no MCTS, no alpha-beta. Reusing the same layers several times on the
  current position (a deeper pass, item 23) is allowed: it never looks at any other position.
- **Network only:** no opening book and no tablebase probing at play time. Books, tablebases and
  Stockfish may be used to make *training labels*.
- **Compact:** ≤ 14M parameters and ≤ 30 MB shipped (≤ 5 MB with int8 is the stretch goal). The owner
  raised the cap from 10M for v4 ("bigger but not too big"); older notes below still say 10M.

Targets (on our scale: Stockfish 17.1 `UCI_LimitStrength`, 0.1 s/move):

| milestone | target |
|---|---|
| v3 | clearly beat v2 in 120-game matches (**done**: 92% head-to-head, 49% vs SF 2000) |
| v4 | score 50% against SF `UCI_Elo` 2000 (v2: 12%, v3: 49%) |
| long term | approach DeepMind's 9M searchless model (~2000 Elo vs bots, see below) |
| stretch | beat DeepMind's 270M model head-to-head with ≤ 10M params and free compute (see "Stretch goal") |
| moonshot | 3000 on our scale, still ≤ 10M params and no search (see "Road to 3000") |

## Where we are

| | v1 | v2 | v3 (shipped) |
|---|---|---|---|
| params | 0.88M | 2.9M | 6.7M (d=256, 8 layers) |
| positions seen in training | ~0.4M | ~2M | ~67M |
| trained on | 2 CPU cores, 1 h | 4 CPU cores, ~4.5 h | Kaggle 2×T4, 5 h |
| puzzles (bench, 1,050) | 10.5% | 39.1% | 59.3% |
| held-out win prob lost / move | 0.086 | 0.054 | 0.028 |
| vs random (120 games) | +19 =99 −2 | +106 =14 −0 | +118 =2 −0 |
| vs SF 1320 (120 games) | 12% | 52% | 92% |
| vs SF 1600 / 2000 | – | 26% / 12% | 73% / 49% |
| round-robin (100 games/pair) | 18% vs v2, 4.5% vs v3 | 82% vs v1, 8% vs v3 | 95.5% vs v1, 92% vs v2 |

Performance rating on our scale: v1 ≈ 975, v2 ≈ 1350–1650, v3 ≈ 1740–1990 (1991 vs SF 2000).

Time-control check: `UCI_Elo` is calibrated at 60s+0.6s, our matches use 0.1 s/move. v3 vs SF 2000
at **1.0 s/move**, two 60-game runs: +23 =17 −20 (52.5%, perf 2017) and +22 =16 −22 (50.0%), vs 48.7%
at 0.1 s. No inflation, so the 0.1 s figures stand (`games/v3_vs_sf2000_1s.pgn`, `_run2.pgn`).

## How we decide (experiment protocol)

- **Benchmarks** (`evaluate.py`, all policy only):
  - `bench/puzzles.csv`: 1,050 Lichess puzzles, 150 per rating band.
  - `bench/heldout.pkl`: 1,000 positions, every move graded by SF 17.1 at depth 12.
  - Matches vs SF `UCI_Elo` 1320 / 1600 / 2000 (and random), 120 games each, paired openings.
  - *New:* the DeepMind paper's 10k Lichess test puzzles (evaluation only), so we can compare
    with the paper directly.
- **Rules:**
  - Compare candidates at **equal GPU time**.
  - A change ships only if it wins **120-game matches**. Held-out loss alone never decides
    (in v2 it looked fine while real games were broken).
  - Every evaluation is appended to `bench/results.jsonl`.
  - One change at a time where possible. v2's "round 2 + polish" changed three things at once
    and we still don't know which one hurt.

## Compute and data

**Compute.**
- Kaggle: 30 GPU-hours/week on 2×T4 (quota resets weekly), sessions up to ~12 h, CPU
  notebooks don't use the GPU quota. `kaggle_run.py` pushes and collects jobs.
- This cloud machine: 4 CPU cores, used for Stockfish grading, evaluation and orchestration.

| measured | speed |
|---|---|
| train 2.9M model, 4 CPU cores | ~200 positions/s |
| train 2.9M model, 1×T4 fp16 | ~3,400 positions/s |
| train 6.7M model, 1×T4 fp16 / with compile / **2×T4** | 1,900 / 2,250 / **3,580** positions/s |
| T4 with bf16 | ~3× slower than fp16 (emulated) |
| Stockfish 17.1, every move graded, depth 8 | ~34 positions/s per core (lean UCI reader) |
| `encode()` | 0.35 ms/position per core |
| batch building (CPU side) | ~700k positions/s, never the bottleneck |

**Data we have** (private Kaggle datasets `yunus0901/vibefish-v2-data`, `yunus0901/vibefish-v3-data`):

| set | positions | labels |
|---|---|---|
| L | 6M | Lichess eval DB, deep evals, top 1–5 moves |
| L2 | 20M | same, top 8 moves kept |
| S_0, S_puz | 220k | every move, SF 17.1 depth 8 |
| S2 (a–h) | ~3.2M | every move, depth 8, puzzle-dump positions (Kaggle CPU + local) |
| D1, D2 | 154k | self-play (DAgger) with move history, every move, depth 8 |
| heldout | 1,000 | every move, depth 12 (benchmark, never train) |

**ChessBench** (DeepMind, `google-deepmind/searchless_chess`; positions CC0 from Lichess, move
grades CC-BY 4.0). **Used as training data, with credit** (owner's call: faster progress is worth
one credit line). Any shipped model trained on it says so in the README: "Trained in part on
ChessBench (DeepMind, CC-BY 4.0)".
- 10M Lichess games, ~530M positions, **15.3 billion** action values: every legal move graded by
  Stockfish 16 at 50 ms per move.
- Action-value training data: 2,148 shards, ~1.1 TB in total (~0.5 GB and roughly 250k
  positions per shard). Stored in `.bag` files (their `bagz.py` reader).
- Also: state-value and best-move sets (~35 GB each), a test set, 10k test puzzles, and
  pretrained 9M / 136M / 270M checkpoints. **We never use their weights**; the checkpoints are
  only opponents (item 20).

## What we learned (keep these)

1. **Labels are everything.** VibeFish can't be stronger than its teacher. v1→v2's +400 Elo
   came mostly from more and better Stockfish labels.
2. **Every input feature must be exercised by training data.** Lichess and puzzle positions
   have no move history, so the history features stayed untrained and real games fell apart
   (0–20 vs SF 1320) until self-play games trained them.
3. **Mate distance belongs in the labels.** Grading mate in *n* as `1 + 0.2/n` lets the bot
   finish games without outside help.
4. **Self-play (DAgger) is the most valuable data per position**, but each round has to prove
   itself in matches. Round 2 + polish in v2 made it worse.
5. **Hand-made features pay off when compute is small** (attack counts, cheapest attacker, pins).
6. **Never run CPU jobs next to multi-threaded training:** one busy core halves PyTorch speed.
7. **On T4s use fp16, not bf16**, and use both GPUs. `torch.compile` and multi-GPU crash together.
8. **Small datasets repeated many times overfit models of ~7M+ params** (we saw the risk; the
   DeepMind paper confirms it).

## Reference: DeepMind, "Grandmaster-level chess without search" (arXiv 2402.04494)

Same idea at ~1,000× our training volume: transformers trained on ChessBench, no search.

| model | Elo vs bots | puzzles (their 10k) | training |
|---|---|---|---|
| **9M** | **~2025** | 88.9% | 10M steps × 4,096 positions (~41B examples), 128 TPUs |
| 136M | ~2260 | 94.5% | same |
| 270M | ~2300 (2895 vs humans, blitz) | 95.4% | same |

What we take from it:
- **Small models can be strong.** 9M params reached ~2000 Elo. Our limit is training volume
  and label quality, not size.
- **Depth matters:** accuracy rose with layers up to ~16; 2–4 layers were poor.
- **Action values win with enough data:** predicting a win probability for *every* move beat
  imitating the best move once data was plentiful.
- **Binned value loss (HL-Gauss, 128 bins)** beat log-loss and L2.
- **Their weak spots are our strengths:** no history in the input (they patch repetitions with
  a rule) and all mates = 100% (they let Stockfish pick the move when winning). VibeFish solves
  both inside the network.
- **Our edge must be efficiency per example:** hand-made features, mate-aware labels,
  self-play, plus as much ChessBench as the free GPU hours can absorb.

## Roadmap

### v3: scale up on Kaggle (done, shipped)

- [x] GPU path on Kaggle: fp16 on T4, both GPUs, shard-wise sampling (RAM), multi-dataset jobs
- [x] Data: +20M Lichess positions (L2), ~3.2M every-move-graded puzzle positions (S2)
- [x] Stage 1: 6.7M model, 3 h on 2×T4 (L + L2 + S + D), 39M positions seen. Puzzles 56.0%,
      held-out wp lost 0.024. 120-game matches: random +114 =6 −0; SF 1320 86.7% (perf 1645);
      SF 1600 64.6% (perf 1704); SF 2000 40.0% (perf 1930; v2: 11.7%).
- [x] Self-play D3: 3,000 games from the stage-1 model, ~218k positions; its own moves lose
      0.027 wp on average (v2's D2: 0.036)
- [x] Stage 2: fine-tune from stage 1, 2 h on 2×T4, 28M positions (S2_e–h 35%, D3 17%, L 15%,
      L2 20%, S/D1/D2 13%; lr 3e-4). Puzzles 59.3% (+3.3), held-out wp lost 0.028 (worse than
      stage 1's 0.024: fewer Lichess positions in the mix + the lr restart). Matches: random
      +118 =2 −0; SF 1320 91.7%; SF 1600 73.3%; SF 2000 48.7% (perf 1991). Better in every
      match, so **stage 2 shipped as v3** (`models/vibefish_v3.pt`).
- [x] Round-robin, 100 games per pair: v3–v2 +86 =12 −2, v3–v1 +91 =9 −0, v2–v1 +64 =36 −0.
- Not used yet: S2_a–d (1.6M positions graded on Kaggle CPU, kernel `vibefish-grade-1`; it
  finished 3 min after stage 2 launched). Attach it with `--kernel-source` next time.
- Lesson for the next fine-tune: keep ≥ 50% Lichess positions in the mix and start at a lower
  lr (≤ 1e-4), so it keeps stage 1's accuracy on ordinary positions.

### v3.1: ChessBench, the biggest expected gain

**Status (in progress).** Finding: the action-value shards are shuffled across all 2,148 files
(100k records of shard 0 share only 8 positions), so a few shards can't give positions with every
move graded. Getting those would need one pass over all 1.1 TB. The behavioral-cloning file is used
instead: 527M positions, each with Stockfish 16's best move (50 ms/move).
- [x] `data.py chessbench`: range-reads any slice of the 36 GB file (no full download) and writes
      records with an UNKNOWN score: a one-hot policy target, skipped by the value and
      action-value losses. Benchmark puzzle positions and held-out positions
      (`bench/heldout_fens.txt`) are excluded. ~4k positions/s on 4 cores, ~480 B/position.
- [x] Kaggle CPU notebooks `vibefish-cb-1` / `-2`: positions 0–20M and 20–40M.
- [x] Equal-GPU-time test, both from v3, 4 h, lr 2e-4, same data otherwise: `vibefish-v31-control`
      (L 25%, L2_00–09 35%, S2_a–h 20%, D3 12%, S/D1/D2 8%) vs `vibefish-v31-cb` (CB1 35%,
      L 10%, L2_00–09 15%, rest the same). L2 is halved in both so CB fits in the 29 GB of RAM.
- [x] Matches vs v3. **Neither ships**; both kept in `models/`:

      | | held-out wp lost / top1 | puzzles | vs v3 (100 g) | SF 2000 (120 g) |
      |---|---|---|---|---|
      | v3 | 0.028 / 52.2% | 59.3% | – | 48.7% |
      | v3.1 control | 0.0252 / 52.9% | 58.8% | 50.5% | 46.7% |
      | v3.1 ChessBench | 0.0243 / 54.3% | 59.1% | 50.0% | 47.5% |

      Lesson: at 6.7M params v3 is saturated. More data (even 18M new ChessBench positions)
      improves held-out accuracy a little but not play. Capacity is the limit, so v4 goes bigger.

Original plan:

1. **Converter.** Read one action-value shard, check the record layout (FEN, move, win
   probability), group records by position, and write our records `(fen, [], {uci: score})`.
   Win probabilities convert back to the centipawn scale with the inverse Lichess curve.
   ChessBench maps every mate to 100%, so mate distance is lost. Keep relying on our self-play
   data (and later tablebases) for finishing games.
2. **Run it on Kaggle CPU notebooks** (free, with internet). Each converts a block of shards into
   our `.npz` format as notebook output; training jobs attach those outputs (`--kernel-source`).
   Disk and output limits decide the block size (~20 GB of output per notebook).
3. **Size the slice to the GPU budget.** At ~3,600 positions/s, a 6 h run sees ~80M positions.
   About 50–100 shards (12–25M positions with every move graded) is the useful amount per run.
4. **Keep grading our own positions too** (Kaggle CPU notebooks, ~5–6M positions per 12 h,
   as `vibefish-grade-1` does): self-play and puzzle positions with mate distance, which
   ChessBench lacks. Kaggle hides a running notebook's log and outputs, so grading jobs must
   be short (≤ 3–4 h) or split into separate notebooks; `vibefish-grade-1` ran 6 h+ with no way
   to tell progress from a hang.
5. **Train** the v3 recipe + ChessBench at equal GPU time vs v3; also evaluate on ChessBench's
   10k puzzles. If it ships, add the README credit line.

### v3.2: network and loss

5. **Depth sweep at fixed size:** d=256×8 vs d=192×14 vs d=160×20 (~6–7M params each).
6. **Geometry-aware attention:** a learned bias per square pair and head, optionally
   content-dependent like Leela's "smolgen". Likely the biggest architectural win per parameter.
7. **Binned action-value head (HL-Gauss, 128 bins)**, then retest playing by action value instead
   of policy now that most data grades every move.
8. **Stronger hand-made features:** static exchange evaluation (does capturing on this square
   win material?), attackers by piece type, king-zone pressure.
9. **EMA of weights** (ship the averaged weights).
10. **Under-promotions** as their own moves.

### v3.3: better labels and self-play

11. **Self-play every version:** varied starts (`--start-puzzles`), deeper labels (depth 12) for
    self-play positions, one change per round, kept only if matches improve.
12. **Endgame tablebases (Syzygy 3–5 pieces, ~1 GB)** as exact labels for endgame positions.
    Fixes unconverted won endgames (part of v2's 14/120 draws against a random mover).
13. **Hard-example mining:** oversample positions where the bot's move loses the most.
14. **Horizontal-flip augmentation** for positions without castling rights.
15. **Analyse the draws against the random mover** (stalemate? repetition? 50-move rule?).

### v4.1: one more self-play round (shipped 2026-11-08)

D5: 4,000 games from the v4 main EMA, every move graded at **depth 12** (B1 + B2), 311k positions,
~2 h per 500 games per CPU notebook. Fine-tune `vibefish-v41-ft`: 150 min from v4's EMA, lr 5e-5,
D5 25% / D4 10% / CB4 20% / L, L2_1*, S2, D3, S_puz, D1, D2; 18.4M positions. `models/vibefish_v4_1.pt`
= its EMA. Mix 2, 200 games: **56.8% vs v4** (+80 =67 −53), **SF 2000 64.0%** (2100; v4 56.2%),
SF 2400 33.0% (= v4). Puzzles 64.3%, held-out 0.0227. 120 games: SF 1600 84.6%, SF 1320 95%,
random +117 =3 −0. Lesson: self-play with deeper labels still pays; next is D6 from v4.1.

### v4: bigger and deeper (shipped 2026-11-07)

**Shipped:** `models/vibefish_v4.pt` = the fine-tune's EMA (`vibefish-v4-ft`: 3 h from the main
run's EMA + `opt.pt`, lr 1e-4, CB4 30%, D4 15%, L2_10–19, S2, D3; 22.6M positions, no slowdown),
played with `--mix 2`. All 200 games: **66.5% vs v3** (+106 =54 −40), **61.5% vs v4-main** at mix 2
(+86 =74 −40), **SF 2000 56.2%** (+86 =53 −61, perf 2044; v4-main 56.0%), **SF 2400 33.0%**
(+38 =56 −106, 2277; v4-main 29.0%, 2244). 120 games: SF 1600 79.6% (1836), SF 1320 96.7%, random
+117 =3 −0. Puzzles 63.5% (v4-main 60.2%), held-out 0.0231 / 54.7%. The D4 self-play data shows
mostly in head-to-head and puzzles; against SF 2000 it is even with v4-main.

v3 grown to **16 layers, d=256, 13.6M params** (owner raised the cap to 14M), initialised from
v3.1-cb with the new layers as identity. **Free Kaggle tier only** (owner, 2026-11-06): ~18 GPU h
left this week, 30 h/week after; CPU notebooks are free. No search, one forward pass.

Everything that can make v4 as strong as possible, roughly by expected gain per cost.
[GPU] = uses GPU quota, [CPU] = free Kaggle CPU notebook or this machine, [code] = change needed.

**A. Already running**
- [x] Pilots, 110 min each on 2×T4, grown from v3.1-cb, CB2 35% of the mix (`models/vibefish_v4_pilot_{a,b}.pt`,
      logs in `bench/train_logs/`):

      | | A: 16 layers + bias | B: 16 layers | v3.1-cb (start) | v3 |
      |---|---|---|---|---|
      | positions seen (speed) | 13.6M (2,050/s) | 14.9M (2,260/s) | | |
      | held-out wp_lost / top1 | 0.0248 / 54.0% | 0.0239 / 54.5% | 0.0243 / 54.3% | 0.028 / 52.2% |
      | puzzles | 60.0% | 59.8% | 59.1% | 59.3% |
      | vs v3, 100 games | **53.0%** (+40 =26 −34) | 48.5% (+33 =31 −36) | 50% | |

      Both within noise of their start. **Why:** the 8 added layers barely trained. Their
      out_proj + linear2 weight norms reached ~3 (v3's layers: 66–86), and the bias reached a
      mean |b| of 0.011. A goes on (bias costs 10% speed, scored best). Fix: item 10.

**B. Data and labels (biggest proven lever: "labels are everything")**
1. **Self-play round D4 from the pilot winner** (DAgger, every move graded at depth 8, varied
   starts with `--start-puzzles`). v2 and v3 got their biggest jumps from this. Run on CPU
   notebooks while the main run trains. [CPU] *Started:* `vibefish-d4-1` / `-2`, 2,000 games
   each from pilot A, 500-game chunks `D4_10`–`D4_23`, start puzzles from row 4.1M on (dagger
   used to take the first rows, where the puzzle benchmark comes from; `--start` now sets it).
   Goes into a fine-tune after the main run.
   *Done:* 4,000 games, 311k positions, own moves lose 0.0247 wp (D3: 0.027). **D5 started**
   (`vibefish-d5-1` / `-2`): 4,000 games from the v4 main EMA, graded at **depth 12** (B2 for the
   positions v4 actually reaches), outputs `D5_10`–`D5_23`.
2. **Deeper labels where it matters:** regrade at depth 12–16 the positions where the model's
   move loses > 0.05 win prob (from D4 and held-out-style samples). Depth 8 is our ceiling now. [CPU]
3. **More fresh ChessBench:** convert positions 40–80M (`cb-3`, `cb-4`) so the main run never
   repeats data; the v3.1 lesson is that repeats don't help. Best-move-only, so keep it ≤ 35% of the mix. [CPU]
4. **Endgame tablebases (Syzygy ≤ 5 pieces, ~1 GB):** exact labels for endgame positions from
   self-play, so won endgames get converted (item 12). [CPU, code]
5. **Hard-example mining:** oversample positions where the model's move loses the most (item 13). [code]
6. **Keep ≥ 50% every-move-graded data** (L, L2, S2, D) in the mix: those train the action-value
   and value heads; ChessBench only trains the policy.

**C. Training recipe**
7. ~~DDP instead of `nn.DataParallel`~~ dropped: v3 measured 1,900 pos/s on one T4 and 3,580 on
   two, so DataParallel already scales 1.88×; DDP could add ≤ 6%.
8. [x] **EMA of weights** (item 9): `train.py --ema 0.999` (default) saves `ema.pt` next to
   `model.pt` and logs `ema_policy_lost` / `ema_policy_top1`. Ship whichever wins.
9. [x] **Optimizer state saved** (`opt.pt`) with every checkpoint; continue a run across
   sessions with `--init model.pt --opt-state opt.pt`.
10. [x] **Learning rate for grown layers:** the pilots' new layers did not move enough (A above).
    `train.py --boost-from 8 --boost 5` trains layers 8–15 and the attention bias at 5× lr
    (1e-3 at lr 2e-4, v3's from-scratch lr). Used for the main run.
11. [x] **Main run 10–11 h** from the pilot winner, with the cosine schedule over the full run and
    a lower final lr. Expect ~60M positions; memmap shards are only needed past ~50M in RAM. [GPU 10 h]
    *Done* (`vibefish-v4-main`, `models/vibefish_v4_main.pt` + `_ema.pt`): from pilot A, boost 5
    on layers 8–15, CB3 35% of the mix, 63.2M positions. Held-out (train.py) 0.0222 → 0.0211,
    EMA 0.0209. The new layers' weight norms grew from ~3 to 16–37, bias |b| 0.011 → 0.18.
    200 games vs v3: model **53.7%** (+73 =69 −58), EMA **55.2%** (+74 =73 −53). Puzzles 59.4% /
    59.1%, evaluate.py held-out 0.0233 / 0.0232 (v3 59.3%, 0.028).
    SF 2000, 120 games (EMA): **47.9%** (+42 =31 −47, perf 1986), even with v3's 48.7%. The head-to-head
    gain doesn't show against Stockfish yet, so the gate isn't met.
    **Unexplained slowdown:** ~2,330 pos/s until step 60,000 (minute 440), then only 1,738 steps
    in the last 193 min, with no error in the log. So the lr tail below 5e-5 never ran.
    The fine-tune logs every 2,000 steps to find where it starts.
12. **Binned value / action-value heads (HL-Gauss, item 7):** DeepMind's best loss. Try in a
    second pilot only after the main run, one change at a time. [code, GPU]

**D. Play-time, still no search**
13. **Tune `--mix`** (policy + action-value blend) on held-out and a 100-game match; v4's
    action-value head is trained on far more every-move data than v2's was. [CPU]
    *Held-out / puzzles, v4 main EMA:* mix 0: 0.0232 / 59.1%; 0.5: 0.0234 / 59.4%; 1: 0.0235 / 59.9%;
    **2: 0.0229 / 60.2%**; 4: 0.0235 / 60.3%; 8: 0.0243 / 59.6%; 16: 0.0247 / 59.1%.
    *Matches (200 games each; same game numbers get the same 4 random opening plies, so the SF rows share openings):*

    | v4 main EMA | vs v3 | vs SF 2000 (perf) |
    |---|---|---|
    | mix 0 | 55.2% (+74 =73 −53) | 52.2% (+85 =39 −76, 2016) |
    | **mix 2** | **60.5%** (+88 =66 −46) | **56.0%** (+90 =44 −66, 2042) |
    | v3 (mix 0) | | 47.2% (+72 =45 −83, 1981) |

    Mix 2 vs mix 0 head-to-head: 51.7%. The action-value head helps most against Stockfish.
    `play.py --mix` added. **Ship with mix 2.** The earlier 120-game 47.9% for mix 0 was noise.
14. **Mirror averaging** for positions without castling rights (left-right flip, average the
    two policies): two passes of the same network on the same position, allowed by the rules (item 18). [code]

**E. Measuring properly**
15. **200-game matches** for v4 vs v3 (100 games has ±5% noise, too much for small gains), plus
    the usual 120-game SF 1320/1600/2000 matches. Ship only if it clearly beats v3. [CPU]
16. [x] **Blunder report by game phase** (opening / middlegame / endgame, lost win prob per move from
    the match PGNs) to aim the next data at the weakest phase. [code, CPU]
    *Done:* `bench/blunders.py` (depth 12, skips the 4 random opening plies). v4 main EMA vs SF 2000,
    120 games: opening 12% of the lost win prob (0.0175/move), **middlegame 70%** (0.0289/move,
    258 moves losing > 0.1, 42 losing > 0.3), endgame 18% (0.0144). v3 vs SF 2000 at 1 s looks
    the same (12% / 74% / 15%, middlegame 0.0271/move). So the weak spot is middlegame tactics.
    Tablebases (B4) can wait; deeper labels on middlegame positions (B2, D5 at depth 12) come first.
17. **DeepMind's 10k test puzzles** for a direct comparison with their 9M / 270M models. [CPU]
18. **Lichess bot account** (owner creates it) for a human-comparable rating (item 19).

**Suggested order:** pilots → (B1, B3 on CPU in parallel; `vibefish-cb-3` / `-4` = positions 40–80M started) → main run with D4 + CB3 in
the mix → evaluate (E15–E17) → tune `--mix` and mirror averaging → ship if it beats v3 →
second-round fine-tune with deeper labels and tablebases if GPU hours remain.

### Later: distillation and compactness

16. **Teacher → student (our own teacher only):** train a bigger VibeFish teacher (e.g. 50–100M)
    on Stockfish labels, have it label positions the small model reaches, and train the small
    model on those labels. Two reasons: it removes the Stockfish CPU bottleneck for self-play,
    and a network's outputs are easier for a small network to copy than search results.
    Never DeepMind's models as a teacher (see "Ruled out").
17. **int8 quantization** of the shipped model (~4× smaller); ship only if matches don't drop.
18. **At play time, still no search:** mirror averaging for positions without castling rights;
    a small ensemble only if the gain justifies the size.
19. **Lichess bot account** (needs the owner to create it) via `play.py --uci` + lichess-bot, for a
    real, human-comparable rating.

### Stretch goal: beat DeepMind's 270M with far less

Their recipe is huge data with a generic model and generic training. Ours has to win on
**efficiency per example**. Smolgen (6) and tablebases (12) are already above. What's new:

20. **Head-to-head test harness.** Load their published 270M (and 9M/136M) checkpoints (run
    here as an opponent only, never shipped or trained on) and play
    120+ game matches VibeFish vs DeepMind, same time control, same rules, both searchless. Also
    run both on their 10k puzzles. This replaces cross-scale Elo guessing with a direct result,
    and it is the gate for every step below. Needs a JAX → inference wrapper and a CPU/GPU cost test.
21. **Auxiliary targets that teach lookahead without search.** Stockfish already computes these
    while grading, so the labels are free: the opponent's best reply, the eval 2–3 plies along
    the PV, hanging pieces and threats per square. Extra heads, used only in training. KataGo
    reported ~50× less compute than AlphaZero-style training, partly from auxiliary targets.
    Test: equal GPU time with vs without, on held-out and matches.
22. **Label quality beats label count where it matters.** Our depth-8 labels are *weaker* than
    theirs (SF16 at 50 ms per move). Grade critical positions (close top-2 moves, big eval swings,
    self-play blunders) at depth 14–18, keep depth 8 for the bulk, tablebases for endgames.
23. **Deeper pass by reusing layers (recurrent depth).** Run the same block 2–4 times on the
    current position: more computation per move, no extra parameters, still no search.
    Also test whether running more loops at play time than in training helps.

Stages: v3 → ~1,700–1,900 (estimate); v3.1–v3.2 + items 21–22 → aim at their 9M (~2025);
v4 + item 23 and more self-play → head-to-head vs their 136M, then 270M.

### Road to 3000 (≤ 10M params, no search)

**Honest ceiling.** No searchless model is known to be at 3000. DeepMind's 270M (27× our cap)
reached ~2895 against humans in Lichess blitz; their 9M ~2025 against bots. 3000 at ≤ 10M would
be a breakthrough, so this is the moonshot; the realistic path is the low-to-mid 2000s first.
The size cap is on the **shipped** model; a bigger VibeFish teacher used only in training is fine.

Levers, biggest first (item numbers refer to the lists above):

| # | lever | why it helps a small model |
|---|---|---|
| 1 | **Training volume:** ChessBench (v3.1) + our own grading, 1B+ examples | their 9M saw ~41B, v3 ~40M. Free GPUs process ~390M positions/week |
| 2 | **Training throughput** (new item 24) | profiling, bigger batches, DDP, shorter inputs: aim for 2× positions/s, i.e. 2× data per GPU hour |
| 3 | **Reusing layers** (item 23) | a 10M model looped 3–4× has the compute of a 30–40M one, no extra params |
| 4 | **Chess-shaped attention** (item 6) + depth sweep (item 5) | Leela's smolgen gave large gains per parameter |
| 5 | **Lookahead targets** (item 21) | predicts what search would find (best reply, eval a few moves on, threats) |
| 6 | **Label quality** (items 12, 22) | deep labels on critical positions, tablebases for endgames, mate distance kept |
| 7 | **Self-play every version** (items 11, 13) | trains where its own blind spots are; hard examples replayed more |
| 8 | **Our own big teacher** (item 16) | 100M VibeFish teacher on Stockfish labels → distil into the ≤ 10M model |
| 9 | **Game memory** (new items 25–26) | plans that last longer than 4 half-moves |
| 10 | **Play-time tricks without search** (item 18) | mirror averaging; a 2–3 model ensemble totalling ≤ 10M |
| 11 | **Don't forget** (new item 27) | replay buffer of older data in every run |

New items:

24. **Faster training:** profile a Kaggle step (data loading vs GPU), confirm the attention runs
    on the fused kernel, try larger batches with LR scaling and feeding both GPUs better than
    `nn.DataParallel` (DDP), and check whether the 68 tokens can shrink. Goal: ≥ 2× positions/s
    at equal results.
25. **Longer move history:** mark the last 8–16 half-moves instead of 4. Needs self-play data to
    train it (lesson 2) and an `ENCODING_VERSION` bump.
26. **Plan memory (recurrent state):** after each move the network keeps a small vector and reads
    it back on its next move, like a notepad of its own thinking ("attacking the king side").
    It never looks at other positions, so it is memory, not search. Trained on whole self-play
    games (truncated backprop through time). A/B test vs the same model without it.
27. **Replay buffer:** every run mixes in a fixed share of older data and the best older self-play
    rounds, so new data never pushes out old skills.

Rough goals on our scale (estimates; each step must win matches): v3 ~1,850–1,950 (actual ~1,740–1,990) · v3.1 ~2,100 ·
v3.2 ~2,300 · v4 ~2,500 · beyond: 2,600+ is the frontier, 3000 the moonshot.

### GPU budget (30 h/week)

| use | hours |
|---|---|
| main training run per version | 6–10 |
| fine-tunes after self-play rounds | 2–4 |
| sweeps (depth, architecture), ~1–2 h each | 4–8 |
| reserve (crashes, reruns) | ~5 |

CPU work (Stockfish grading, self-play, evaluation) runs on Kaggle CPU
notebooks and this machine, never on the GPU clock.

## Ruled out

- **Opening book** and **tablebase probing at play time**: not search, but not the network either.
- Anything that evaluates positions after a candidate move (1-ply lookahead, MCTS, alpha-beta).
- Training on the benchmark sets (`bench/exclude.txt` keeps puzzle positions out of training data).
- **DeepMind's models in any role but opponent** (owner's call): never as a teacher (a
  searchless teacher can't beat Stockfish labels) and never their weights in our model. Their
  checkpoints are only played against (item 20). ChessBench *data* is allowed, with credit.

## History

**v1 → v2 diagnosis.** v1 had no move history in its input (it shuffled into repetitions),
squashed every mate to 1.0 (never finished games), trained on 80k depth-8 positions with a
constant learning rate, and only knew attack counts. v2 fixed each one.

**v2 build (4 CPU cores).**
- Data: 6M Lichess positions, 220k every-move-graded positions; python-chess's `analyse()` was
  replaced by a bare UCI reader (2× faster grading).
- Pilots: at equal time, 2.9M params tied 1.3M params with 42% fewer samples, so we picked the
  bigger one.
- Main run: puzzles 25% but 0–20 vs SF 1320 because the history features were untrained;
  stripping history at play time gave +3 =4 −13 (perf ≈ 1130).
- The machine rebooted at step ~2400. Continued from step 2000 with self-play data D1 in the
  mix; after 128k samples: +6 =2 −12 vs SF 1320.
- Final (shipped): numbers in "Where we are". Policy alone beat policy + action-value mix in
  games (0.31 vs 0.25 at SF 1600).
- Self-play round 2 + a 30-min polish: worse in games (perf 1340 vs 1417), not shipped.

**v3 so far (Kaggle).**
- Smoke tests: T4 reports bf16 "supported" but emulates it; fp16 is ~3× faster. `torch.compile`
  +17%, two GPUs +86%, both together crash.
- `train.Source` now samples shard by shard (no full concatenate) so 26M positions fit in 29 GB RAM.
- L2 (20M positions) encoded on 4 cores in ~35 min; uploads run at ~28 MB/s.
- Stockfish grading of puzzle positions runs both on a Kaggle CPU notebook and locally.
- Stage 1 (3 h) and stage 2 (2 h) on 2×T4; D3 self-play locally. The machine rebooted twice
  during D3, so it ran as 500-game chunks (a shell loop that skips finished files).
- Final: stage 2 beat stage 1 in every match and v2 92% head-to-head (numbers in the v3 roadmap).

# VibeFish v2 plan

Goal: a **small, searchless** chess bot. One forward pass picks the move. No alpha-beta,
no MCTS, no lookahead at play time. Strength has to come from the network.

## Where v1 stands (baseline)

0.88M-param transformer, 80k Stockfish-11 depth-8 positions, ~1,500 gradient steps.
Policy-only: draws 18/20 vs a random mover (repetitions), loses 17/20 to SF11 Skill 0.
Top-1 agreement with Stockfish 25%.

## Diagnosis: why v1 is weak

| # | problem | where | effect |
|---|---|---|---|
| 1 | Input has no history: no repetition count, no 50-move clock, no last move | `encode()` | Stockfish labels repeating moves as draws, the model can't see why. Shuffles. |
| 2 | All mates map to win prob 1.0 (`mate_score=10000` → sigmoid saturates) | `teacher.grade()` | Mate in 1 and mate in 15 look identical, so it never learns to finish. |
| 3 | Tiny, shallow data: 80k positions, depth 8, SF 11 | `loop.py` | Noisy labels, massive underfitting. |
| 4 | Constant LR, no schedule | `loop.py` | Leaves accuracy on the table. |
| 5 | Features: attack *counts* only | `encode()` | Can't tell a pawn guard from a queen guard, doesn't see pins. |

## Budget (measured on this box: 4 CPU cores, no GPU)

| thing | speed |
|---|---|
| train 0.8M transformer (fp32, batch 512) | ~320 positions/s |
| train 2.7M transformer | ~135 positions/s |
| bf16 autocast | slower (no AMX), don't use |
| SF 17.1 grading **all** moves, depth 8 | ~34 positions/s/core |
| Lichess eval DB download | ~12 MB/s ≈ 150k positions/s |

So **training compute is the bottleneck**, not data. Lichess data is free and deep (depth 20–50),
but only grades the top 1–5 moves. Local SF grades every move, which is what teaches blunder
avoidance, and it is affordable at depth 8.

## v2 design

**Input** (68 tokens, each a sum of per-feature embeddings, board from side-to-move view):
- per square: piece, my/their attacker count (0–3), my/their *cheapest* attacker (P..K),
  pinned flag, last-move markers (opponent's last move, my previous move)
- global tokens: castling rights, en-passant file, halfmove-clock bucket, repetition count

**Heads**
- policy: `q_from · k_to` bilinear over 64×64 (as v1), trained with soft targets
  `softmax(grade / tau)` over graded moves
- action value: second bilinear head, `sigmoid` → per-move win prob, BCE on graded moves
- position value: win prob, from mean-pooled tokens

**Grades** (one scalar per move, side-to-move view): `wp = sigmoid(0.00368 * cp)` (Lichess curve).
Mates live *above* 1: mate in n → `1 + 0.2/n`; getting mated in n → `-0.2/n`. Shorter mates
win the softmax clearly. Value/BCE targets clip grades to [0, 1].

**Data**
- `L`: Lichess eval DB, multi-PV deep evals, top-k moves (millions)
- `S`: SF 17.1 depth 8, all legal moves, on Lichess FENs (no history)
- `D`: DAgger, the v2 model plays games, SF grades every move with full history
  (teaches recovery from its own mistakes, and repetition/50-move behaviour)

**Training**: AdamW, warmup + cosine decay, batches mixed from all sources, pre-encoded numpy
shards so the CPU only does the network.

## Measurement (all policy-only, one forward pass per move)

1. **Puzzles**: Lichess puzzle DB, fixed 1,050-puzzle sample (150 per rating band). Solved = every
   solution move matched (any mate-in-1 counts).
2. **Held-out wp lost**: 1,000 Lichess positions never trained on, every move graded by SF 17.1
   depth 12. Avg win prob given up vs best move, and top-1.
3. **Matches**: vs SF 17.1 `UCI_LimitStrength` at several `UCI_Elo` levels, plus random mover.

## Steps

- [x] 0. Environment: venv, SF 17.1 binary, data download, throughput measurements
- [x] 1. Docs: PLAN.md, CLAUDE.md, requirements
- [x] 2. v2 code: encoding, model, grading fix, data pipeline, trainer, eval, tests
- [x] 3. Baseline: v1 on the new benchmarks (below)
- [x] 4. Data: L = 6M Lichess positions; S_0 = 100k Lichess FENs and S_puz = 119k puzzle positions,
      all moves graded at depth 8
- [x] 5. Pilot runs: model size at equal wall-clock (below)
- [x] 6. Main training run (d=192, 6 layers; stopped by a reboot at ~2400 steps, kept step 2000)
- [x] 7. DAgger round 1 + 150-min continuation (shipped). Round 2 + polish: worse in games, not shipped
- [x] 8. Final eval, README, ship `models/vibefish_v2.pt`

## Log

**v1 baseline** (new benchmarks, policy only): puzzles 10.5%, held-out wp lost 0.086 / top-1 33%,
vs random +1 =18 −1, vs SF 17.1 `UCI_Elo` 1320 +0 =7 −13 (perf ≈ 1050).

**Data notes.** The Lichess eval DB leans to endgames (median 18 pieces, 43% of positions have a
single graded move), so the all-moves set was topped up with puzzle positions (middlegame tactics).
python-chess's `analyse()` spent as much CPU as Stockfish (it validates PVs); a bare UCI reader
doubled grading speed with identical results.

**Pilots** (Lichess data only, 40 min each, 2 threads, run side by side):

| model | params | samples | held-out wp lost | top-1 |
|---|---|---|---|---|
| d=128, 6 layers | 1.29M | 254k | 0.0809 | 31.3% |
| d=192, 6 layers | 2.88M | 148k | 0.0807 | 30.7% |

Tied at 40 min with 42% fewer samples for the bigger net, so it learns more per sample and
should win over a long run. Picked d=192. Policy-only vs policy+action-value mix: no difference
yet. Both already beat v1's held-out score after 40 min.

`torch.compile` would add ~15%, but this box's g++ lacks `cc1plus`, so training runs eager.

**Main run, step 2000 (1M samples):** held-out wp lost 0.066, top-1 38%, puzzles 25%, but
**0–20 vs SF 1320**, worse than v1. Cause: train/play mismatch. Lichess and puzzle positions carry
no move history, so the last-move markers, 50-move-clock buckets and repetition token were never
trained. In real games they are always set, and their untrained (random-init) embeddings scramble
the input. Same checkpoint with history stripped at play time: +3 =4 −13 vs SF 1320 (perf ≈ 1130),
+9 =1 −0 vs random. Fix: DAgger games (which do carry history) go into the fine-tune, so those
features get trained. The DAgger games are generated with history stripped, so they show the
positions the model actually reaches. Lesson: **every input feature needs training data that
exercises it**. A feature that is constant in training is a landmine at play time.

**Restart.** The machine rebooted at step ~2400 of the main run. Continued from the step-2000
weights (lr 5e-4 → cosine over 150 min) with D1 (97k DAgger positions, 20%) in the mix.
After 128k samples, full-history play went from 0–20 to **+6 =2 −12 vs SF 1320** (perf ≈ 1210)
and +19 =1 −0 vs random. The history features work once they see data.

**CPU contention.** One extra busy process (even at nice 10) halves 4-thread PyTorch training,
because every op waits for its slowest thread. Never generate data next to a training run.
Run them one after the other at full width.

**Final (VibeFish v2 = continuation model, policy only, mix 0):** puzzles 39.1%, held-out wp lost
0.054 / top-1 42%, vs random +106 =14 −0, vs SF 1320 +48 =29 −43 (52%), vs SF 1600 +18 =26 −76,
vs SF 2000 +3 =22 −95 (120 games each). v1 on the same 120-game tests: random +19 =99 −2,
SF 1320 +3 =23 −94. Policy-only beat the policy+action-value mix in games (pooled 0.31 vs 0.25 at
SF 1600) and on puzzles, so play uses the policy head alone.

## VibeFish v3 plan (with a GPU)

Goal: keep the shipped bot small (≤ ~10M params, ≤ 20 MB, one forward pass per move) and gain
another few hundred Elo. Stretch: score 50% against SF `UCI_Elo` 2000 (v2: 12%).

### What a GPU changes

v2 was limited by **training compute**: ~200 positions/s on 4 CPU cores, 2M positions seen in
total. A single modern GPU should train the same net one to two orders of magnitude faster
(estimate; step 0 measures it). Then the bottlenecks move:

| bottleneck | why | plan |
|---|---|---|
| labelled data | 6M Lichess positions would be used up in minutes | full Lichess eval DB (~290M positions) + DeepMind's ChessBench |
| Stockfish grading | CPU only, ~34 positions/s/core at depth 8 | grade on every spare CPU core, *not* next to training; later let a big GPU teacher label instead |
| Python encoding | `encode()` takes 0.35 ms/position, single-threaded | encode once, in parallel, into shards; vectorise with numpy bitboards if it's still too slow |
| RAM | `train.Source` loads every shard (~480 B/position) | stream shards from disk (memmap) instead |

### Steps

0. **GPU smoke test.** `train.py` now picks CUDA automatically and trains in bf16 there
   (untested on a real GPU so far). Run 10 minutes of the v2 recipe, record positions/s and
   GPU utilisation. If utilisation is low, the CPU-side batch building is the bottleneck: move
   `to_grade` / padding to the GPU or use a DataLoader with worker processes.
1. **Streaming data loader.** memmap shards, sampled shard by shard, so hundreds of millions of
   positions fit on disk instead of in RAM.
2. **Data scale-up.**
   - Encode the whole Lichess eval DB (22.5 GB compressed) with `data.py lichess` on all cores.
   - Add DeepMind's ChessBench (`google-deepmind/searchless_chess`): positions from real Lichess
     games with *every* legal move graded by Stockfish. That's exactly our "all moves graded" data
     at a far larger scale, and it is middlegame-heavy, which fixes our endgame skew. Check the
     dataset's licence and size first; a slice is enough to start.
   - Neither source has move history, so the DAgger games remain the only teacher for the
     history features (lesson from v2).
3. **Scaling sweep at fixed GPU time** (e.g. 2 h each): d=192/6L (v2, 2.9M), d=256/8L (~6M),
   d=384/8L (~14M). Pick using puzzles and 120-game matches, not held-out loss alone (v2's
   held-out loss looked fine while games were broken).
4. **DAgger at scale.** Several rounds: the current model plays thousands of games, Stockfish
   grades every position (with history), then fine-tune. Run generation on CPU cores that the
   training job doesn't use. Try depth 10–12 labels for the late rounds.
5. **Teacher → student (optional, biggest upside).** Train a large teacher (50–100M params) on
   the same data, then distil it into the small shipped model. The teacher can label millions of
   self-play positions per hour on the GPU, which removes the Stockfish bottleneck for the
   student. The shipped bot stays small and searchless.
6. **Known weaknesses.** Analyse the 14/120 draws against the random mover; support
   under-promotions; horizontal-flip augmentation for positions without castling rights.
7. **Evaluation.** Same `bench/` sets, plus matches at SF 1600 / 2000 / 2400 with 240+ games so
   differences of ~50 Elo are visible. Optional: run it as a Lichess bot through `play.py --uci`
   for a human-comparable rating.

### Which GPU

The available machine is a laptop with **Intel Iris Xe** integrated graphics. That is not enough
for this plan: it shares system RAM, PyTorch reaches it only through the separate XPU backend
(official support for Iris Xe is uncertain), and at best it would be a few times faster than
the CPU. `--device xpu` can be tried, but it's not worth building on.

Plan instead:
- **Training on a free cloud NVIDIA GPU**: Kaggle notebooks (weekly GPU quota) or Google Colab
  (T4). `train.py` picks CUDA automatically: bf16 on Ampere or newer, fp16 + loss scaling on
  T4/P100.
- **The laptop's CPU cores for Stockfish grading** (`data.py sf` / `data.py dagger`), copying
  the shards to the training machine. Grading and training on separate machines also avoids
  the contention problem from v2.
- Free notebooks have limited disk and session time, so start with a slice of the data that
  fits (e.g. the 6M v2 positions plus new DAgger games), and checkpoint often.

Needed to size it: the laptop's CPU core count, and which notebook service you'll use.

## Backlog: getting stronger without search

The rule: the bot may only look at the **current** position (once, or a handful of times), never
at positions after a move. Ordered by expected gain per hour of work. Nothing here is started yet.

### Better teachers (the student can't beat its labels)

1. **Endgame tablebases.** Syzygy 3–5-piece tables (~1 GB; python-chess reads them) give
   exact win/draw/loss and distance for every move in those endgames, better than Stockfish. They also
   target the unconverted won endgames (part of the 14/120 draws vs random). Cheap.
2. **Deeper Stockfish labels.** Depth 8 → 12–14 for the highest-value positions (DAgger games,
   middlegames). Costs ~4–8× CPU per position.
3. **Big teacher → small student.** A 50–100M-param net trained on the same data labels unlimited
   positions at GPU speed; the shipped bot learns from it. Removes the Stockfish throughput limit.

### Smarter use of data

4. **Hard-example mining.** Oversample positions where the bot's move loses the most win probability.
   Most positions it already gets right.
5. **More all-moves-graded middlegames from real games** (the Lichess eval DB leans to endgames;
   puzzles rated 2000+ are at ~9%).
6. **Horizontal-flip augmentation** for positions without castling rights (free extra data).
7. **More varied self-play starts** (started: `--start-puzzles`), and measure every DAgger round
   with 120-game matches before keeping it.

### Chess-aware network

8. **Geometry-aware attention.** A learned bias per square pair (same rank/file/diagonal, knight
   jump), as in Leela Chess Zero's transformer ("smolgen"). Likely the biggest win per parameter.
9. **Fuller history input.** The last few positions (Leela-style), not only last-move markers.
10. **Under-promotions** as their own moves (currently folded into the queen promotion).

### Cheap training tricks

11. **EMA of weights.** Ship an average of recent weights; nearly free, usually a little stronger.
12. **Tuning.** `tau` (target softness), learning rate, run length (held-out loss was still
    falling at the end of every run so far), HL-Gauss / binned value targets instead of BCE.

### At play time, still no search

13. **Mirror averaging.** For positions without castling rights, average the scores of the board
    and its left-right mirror: two looks at the same position.
14. **Small ensembles.** Several nets vote on the current position. Costs speed and size, so only
    if the gain is large ("small and compact" is a goal).

### Ruled out

- **Opening book.** Not search, but not the network either. It breaks the "network only" rule.
- Anything that evaluates positions after a candidate move (1-ply lookahead, MCTS, alpha-beta).

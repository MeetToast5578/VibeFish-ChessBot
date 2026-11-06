# VibeFish plan

## Goal and rules

A **small, searchless** chess bot.

- **Searchless:** the network looks only at the *current* position (once, or a handful of
  times, e.g. a mirrored copy) and picks a move. Nothing evaluates positions after a candidate
  move: no 1-ply lookahead, no MCTS, no alpha-beta. Reusing the same layers several times on the
  current position (a deeper pass, item 23) is allowed: it never looks at any other position.
- **Network only:** no opening book and no tablebase probing at play time. Books, tablebases and
  Stockfish may be used to make *training labels*.
- **Compact:** ≤ 10M parameters and ≤ 20 MB shipped (≤ 5 MB with int8 is the stretch goal).

Targets (on our scale: Stockfish 17.1 `UCI_LimitStrength`, 0.1 s/move):

| milestone | target |
|---|---|
| v3 | clearly beat v2 in 120-game matches |
| v4 | score 50% against SF `UCI_Elo` 2000 (v2: 12%) |
| long term | approach DeepMind's 9M searchless model (~2000 Elo vs bots, see below) |
| stretch | beat DeepMind's 270M model head-to-head with ≤ 10M params and free compute (see "Stretch goal") |

## Where we are

| | v1 | v2 (shipped) | v3 (in progress) |
|---|---|---|---|
| params | 0.88M | 2.9M | 6.7M (d=256, 8 layers) |
| positions seen in training | ~0.4M | ~2M | ~40M (stage 1) + stage 2 |
| trained on | 2 CPU cores, 1 h | 4 CPU cores, ~4.5 h | Kaggle 2×T4, ~5 h |
| puzzles (bench, 1,050) | 10.5% | 39.1% | – |
| held-out win prob lost / move | 0.086 | 0.054 | – |
| vs random (120 games) | +19 =99 −2 | +106 =14 −0 | – |
| vs SF 1320 (120 games) | 12% | 52% | – |
| vs SF 1600 / 2000 | – | 26% / 12% | – |

Performance rating on our scale: v1 ≈ 975, v2 ≈ 1350–1400.

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

### v3: scale up on Kaggle (in progress)

- [x] GPU path on Kaggle: fp16 on T4, both GPUs, shard-wise sampling (RAM), multi-dataset jobs
- [x] Data: +20M Lichess positions (L2), ~3.2M every-move-graded puzzle positions (S2)
- [x] Stage 1: 6.7M model, 3 h on 2×T4 (L + L2 + S + D), 39M positions seen. Puzzles 56.0%,
      held-out wp lost 0.024, +24 =7 −9 vs SF 1600 (40 games, perf ≈ 1737; v2 scored 31%).
- [ ] Self-play round D3 from the stage-1 model (half the games from real middlegame positions)
- [ ] Stage 2: fine-tune with S2 + D3
- [ ] Benchmarks vs v2; ship v3 only if it wins the 120-game matches

### v3.1: ChessBench, the biggest expected gain

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
   ChessBench lacks.
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

### v4: distillation and compactness

16. **Teacher → student (our own teacher only):** train a bigger VibeFish teacher (e.g. 50–100M)
    on Stockfish labels, have it label positions the small model reaches, and train the small
    model on those labels. Only worth it if Stockfish CPU time becomes the self-play bottleneck.
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

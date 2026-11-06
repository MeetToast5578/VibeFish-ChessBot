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
- [ ] 6. Main training run (d=192, 6 layers, 250 min)
- [ ] 7. DAgger round(s) + fine-tune
- [ ] 8. Final eval, README, ship v2 weights

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

## Ideas parked (only if time remains)

- Horizontal-flip augmentation for positions without castling rights
- HL-Gauss / binned value targets instead of BCE
- Distill a bigger model into the small one
- Under-promotion support (currently folded into queen)

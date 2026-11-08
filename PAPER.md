# VibeFish: a small searchless chess transformer, from v1 to v4.1

*A technical report on building a chess bot that picks every move with one forward pass of a
13.6M-parameter network, trained on Stockfish labels with free compute.*

---

## Abstract

VibeFish is a chess engine with **no search**. A transformer looks at the current position once and
outputs a score for every legal move; the bot plays the best-scoring move. There is no alpha-beta,
no Monte Carlo tree search and no lookahead of any depth. All playing strength has to come from the
network. Over five versions we went from a 0.88M-parameter model trained for one hour on two CPU
cores (v1, ~975 Elo on our Stockfish scale, 10.5% of benchmark puzzles) to a 13.6M-parameter model
trained on free Kaggle T4 GPUs (v4.1, performance ~2,100 against Stockfish `UCI_Elo` 2000, 64% of
benchmark puzzles). The gains came from, in order of size: (1) more and better Stockfish labels,
including every-move grades and self-play positions; (2) a board encoding that carries move history,
repetitions and cheap tactical features; (3) scale (parameters, then depth); (4) training details
such as a higher learning rate for newly added layers and weight averaging; and (5) blending the
network's own action-value estimate into the move choice. We also report what did not work: more
training on the same data once a model was saturated, a second self-play round on top of v4.1, and
several held-out improvements that did not translate into won games.

---

## 1. Introduction

Classical engines (Stockfish) and neural engines (AlphaZero, Leela) both combine an evaluation with
a large search: they look many moves ahead. DeepMind's *Grandmaster-level chess without search*
(Ruoss et al., 2024, arXiv 2402.04494) showed that a transformer trained on enough Stockfish data
can play strong chess with no search at all: a 270M model reached ~2,895 Elo against humans in
blitz, and a 9M model ~2,000 against bots, after ~41 billion training examples on 128 TPUs.

VibeFish asks the same question at a very different budget: **how strong can a searchless model
get with a few million parameters and free compute?** The project rules, fixed by the owner from
the start, are:

- **Searchless.** The network sees only the current position. Nothing evaluates the position after
  a candidate move. (v1 had an optional 1-ply lookahead; it was dropped and is now ruled out.)
- **Network only at play time.** No opening book and no tablebase probing. Stockfish, books and
  tablebases may be used to make *training labels*.
- **Compact.** ≤ 10M parameters until v3, raised to ≤ 14M and ≤ 30 MB for v4.
- **Free compute.** A 4-core cloud machine, Kaggle's free tier (30 GPU-hours/week on 2×T4,
  free CPU notebooks), no paid GPUs.
- **DeepMind's models only as opponents.** Their ChessBench *data* is used with credit; their
  weights are never used, and never as a teacher.

---

## 2. Method

### 2.1 Board representation (v2 onward)

The board is always shown **from the side to move's point of view**: when Black is to move, the
board is mirrored, so the network only learns chess "as White". Moves are indexed in that frame as
`from × 64 + to` (4,096 possible moves). Under-promotions are folded into the queen promotion.

The input is a sequence of **68 tokens**: 64 square tokens and 4 global tokens. Each token is a
**sum of learned embeddings**, one per feature, plus a learned position embedding.

| token | features (number of values) |
|---|---|
| each square | piece: empty, 6 mine, 6 theirs (13) · my attackers 0–3+ (4) · their attackers 0–3+ (4) · my cheapest attacker: none/pawn…king (7) · their cheapest attacker (7) · pinned to own king (2) · last-move mark: none, opponent's last move from/to, my previous move from/to (5) |
| global 1 | castling rights, 4 bits (16) |
| global 2 | en-passant file or none (9) |
| global 3 | 50-move clock ÷ 10 (10) |
| global 4 | repetition count of this position: 0, 1, 2+ (3) |

The attacker counts, cheapest attacker and pins are cheap hand-made features (computed with
python-chess in ~0.35 ms per position). They save a small network from having to discover them.
The last-move marks, the 50-move clock and the repetition count are **history features**. They
matter in real games (repetitions, the 50-move rule) and caused v2's biggest bug (§4.2).

v1 used a simpler encoding: 66 tokens, each square one embedding for (piece × my attackers × their
attackers), plus castling and en-passant tokens, and no history.

### 2.2 Network

`ChessGPT` (`model.py`) is a pre-norm transformer encoder (PyTorch `TransformerEncoderLayer`, GELU,
feed-forward width 4·d, no dropout) followed by a final LayerNorm and three heads:

- **Policy head:** two linear maps of the 64 square vectors, `q_from · k_to / √d`, giving a logit
  for every from→to pair (4,096 logits). Illegal moves are masked at play time.
- **Action-value head:** the same shape with its own weights; `sigmoid` of each logit is the
  predicted win probability *after* that move.
- **Value head:** a small MLP on the mean of all tokens; the predicted win probability of the position.

v4 adds a **square-pair attention bias**: a learned additive bias of shape
`[layers, heads, 68, 68]` inside every attention layer, so each head can learn fixed geometric
relations between squares (same diagonal, knight jump, king zone) without spending capacity on
inferring them from position embeddings. At 8 heads it costs 8 × 68 × 68 ≈ 37k parameters per
layer, 0.6M for all 16 layers.

| version | width d | layers | heads | params | file |
|---|---|---|---|---|---|
| v1 | 128 | 4 | 4 | 0.88M | 3.5 MB |
| v2 | 192 | 6 | 6 | 2.9M | 5.8 MB |
| v3 | 256 | 8 | 8 | 6.7M | 13.4 MB |
| v4, v4.1 | 256 | 16 | 8 | 13.6M (with bias) | 27 MB |

Weights are stored in fp16.

### 2.3 Labels: grades, not just best moves

Stockfish scores are converted to **grades**: one number per move, from the side to move's view.

- Centipawns use the Lichess win-probability curve: `wp = sigmoid(0.00368 × cp)`.
- **Mates sit outside [0, 1]**: mate in *n* for the side to move is `1 + 0.2/n`, getting mated in *n*
  is `−0.2/n`. A shorter mate always ranks above a longer one. v1 graded every mate as 1.0, so mate
  in 1 and mate in 15 looked equally good and v1 rarely finished a won game.
- Raw data keeps **Stockfish scores**, not grades, so the curve can change without regenerating data.

### 2.4 Losses

For a position with graded moves `g_i` (best first, padded):

- **Policy loss:** cross-entropy against a soft target `softmax(g_i / τ)` over the graded moves,
  with τ = 0.03 win-probability units. Near-equal moves share the target; clearly worse moves get
  almost none. A ChessBench record (best move only, no score) gives a one-hot target.
- **Action-value loss:** binary cross-entropy between `sigmoid(action_i)` and `clip(g_i, 0, 1)` for
  every graded move.
- **Value loss:** binary cross-entropy between the value head and the best move's clipped grade.

The total loss is the sum of the three. Best-move-only records contribute to the policy loss only.

### 2.5 Data

| set | positions | source | labels |
|---|---|---|---|
| L, L2 | 6M + 20M | Lichess evaluation database | deep Stockfish evals (depth 20–50), top 1–5 / top 8 moves |
| S_0, S_puz | 220k | Lichess positions and puzzle positions | every legal move, Stockfish 17.1 depth 8 |
| S2 (a–h) | 3.2M | Lichess puzzle-dump positions | every legal move, depth 8 (Kaggle CPU notebooks + local) |
| D1–D4 | ~0.7M | **self-play** of v2, v3 stage 1, v4 pilot A | every legal move, depth 8, with move history |
| D5, D6 | 311k + 320k | self-play of v4 main run and v4.1 | every legal move, **depth 12**, with move history |
| CB1–CB4 | 80M | DeepMind ChessBench best-move set (Stockfish 16, 50 ms) | best move only |

**Self-play (DAgger).** The current model plays games against itself (sampling early moves, half the
games starting from puzzle-game positions), and Stockfish grades every legal move in every position
it reached, with the full move history. These are the positions the model actually gets into,
including the bad ones its own mistakes cause, so they teach it to recover. Self-play data is also
the only data with move history, so it is the only data that trains the history features.

**Benchmarks are never trained on.** `bench/exclude.txt` keeps the 1,050 puzzle positions out of every
data set, and `bench/heldout_fens.txt` the 1,000 held-out positions. Self-play start positions come
from puzzle rows far past the ones the benchmark was drawn from.

### 2.6 Training

- **Optimizer:** AdamW (β = 0.9, 0.95; weight decay 0.01), gradient clipping at 1.0.
- **Schedule:** linear warmup, then cosine decay over a fixed wall-clock budget (so runs are compared
  at equal time), down to 5% of the peak learning rate.
- **Mixed sources:** each batch is filled from several data sources by fixed weights
  (`glob:weight[:kmax]`). Each source is sampled without replacement and reshuffled each pass.
- **GPU:** fp16 autocast with gradient scaling on T4s (bf16 is emulated on T4 and ~3× slower);
  both GPUs through `nn.DataParallel`, which already scaled 1.88× on two T4s.
- **EMA of weights** (decay 0.999), saved next to the raw weights. v4 and v4.1 ship the EMA.
- **Depth up-scaling (v4):** v3's 8 layers are copied into a 16-layer model. The 8 new layers start
  as identity: their attention output projection and second feed-forward matrix are zero, so under
  pre-norm each new block adds nothing at first. The grown model plays exactly like v3 before training.
- **Learning-rate boost for new layers (v4):** layers 8–15 and the attention bias train at 5× the
  base learning rate (`--boost-from 8 --boost 5`); see §4.5.

### 2.7 Playing a move

One forward pass gives the policy logits and the action-value logits. Over the legal moves:

```
score(move) = log_softmax(policy)[move] + mix × log_sigmoid(action)[move]
```

The bot plays the highest-scoring move. v1–v3 play with `mix = 0` (policy only). v4 and v4.1 play
with **`mix = 2`**, which weights in the network's own estimate of each move's win probability. Both
heads come from the same single pass, so this is still searchless: nothing looks past the current
position.

---

## 3. Evaluation

| benchmark | what it measures |
|---|---|
| **Puzzles** | 1,050 Lichess puzzles, 150 in each rating band 400–2800+. Solved = every move of the solution found (any mate in 1 also counts). |
| **Held-out** | 1,000 Lichess positions, every legal move graded by Stockfish 17.1 at depth 12. *wp lost* = win probability given up per move compared with the best move; *top1* = the best move was chosen. |
| **Matches vs Stockfish** | Stockfish 17.1 with `UCI_LimitStrength` at `UCI_Elo` 1320 / 1600 / 2000 / 2400, 0.1 s per move. 120 or 200 games; colours alternate on the same 4 random opening plies per game pair. |
| **Head-to-head** | the same paired-opening format between two VibeFish versions. |

**Performance rating** is `elo + 400 · log10(s / (1 − s))` for score *s* against `UCI_Elo` *elo*. It is on
*Stockfish's limited-strength scale*, not Lichess or FIDE. We checked that the short time control
does not inflate it: v3 vs SF 2000 at 1 s per move scored 52.5% and 50.0% in two 60-game runs, versus
48.7% at 0.1 s.

**Decision rule.** A candidate ships only if it wins matches: head-to-head against the current
version and not worse against Stockfish. Held-out loss alone never decides; in v2 it looked fine
while real games were broken.

**Noise.** A 200-game match has a standard error of about ±3.5 points. The same model moved 6 points
between two 200-game runs against SF 2400 (Stockfish at 0.1 s is not deterministic), so close calls
need 400 games.

---

## 4. The versions

### 4.1 v1: Stockfish-graded distillation on 2 CPU cores

v1 (0.88M parameters, 4 layers) was trained by a DAgger loop: in round 0 Stockfish 11 played itself;
in later rounds the model played (with Stockfish stepping in with probability β, falling from 1.0 to
0.2), and Stockfish graded every legal move (MultiPV) in every position reached. The model learned
the **whole grade distribution**, not just the best move. Five rounds, 80k positions, about one hour
on two CPU cores.

**Result:** held-out wp lost 0.086, 33% top-1, 10.5% of puzzles. It drew 99 of 120 games against a
*random mover* and scored 12% against SF 1320 (performance ≈ 975).

### 4.2 v2: diagnosing v1 (4 CPU cores)

Four causes were identified, and each was fixed:

1. **It could not see repetitions.** Stockfish graded repeating moves as draws, but v1's input had no
   history, so those labels were noise. v2 added the repetition token, the 50-move clock and the
   last-two-moves marks.
2. **Every mate looked the same.** v2 grades mate in *n* as `1 + 0.2/n` (§2.3).
3. **Too little data.** v2 trained on 6M Lichess-eval positions, plus ~220k every-move-graded
   positions, plus its own self-play games. A lean UCI reader replaced python-chess's `analyse()`,
   which doubled grading speed.
4. **Richer features:** cheapest attacker on each side and pins.

**The bug that mattered most.** Lichess and puzzle positions have no move history, so the history
embeddings were never trained. In real games they are always set, and the untrained embeddings
scrambled the input. A checkpoint that solved 25% of puzzles lost **0–20** to SF 1320. Ten minutes of
training on self-play games, which do have history, fixed it (then 52% over 120 games after the full
run). Lesson: **every input feature must be exercised by training data.**

A pilot at equal time showed a 2.9M model tying a 1.3M model on 42% fewer samples, so v2 used the
larger one (d = 192, 6 layers). A second self-play round plus a 30-minute polish was *worse* in games
and was not shipped.

**Result:** puzzles 39.1%, held-out 0.054, random +106 =14 −0, SF 1320 52%, SF 1600 26%, SF 2000 12%.
v2 beat v1 82% head-to-head (~+260 Elo).

### 4.3 v3: scaling up on free GPUs

The trainer moved to Kaggle (2×T4, fp16) with shard-wise sampling so 26M positions fit in 29 GB of
RAM. The model grew to 6.7M parameters (d = 256, 8 layers).

- **Stage 1** (3 h, 39M positions): Lichess evals L + L2 (80%), every-move-graded positions (10%),
  self-play D1/D2 (10%); lr 1e-3. Puzzles 56.0%, SF 2000 40%.
- **Self-play D3:** 3,000 games from the stage-1 model, 218k positions; its own moves lost 0.027 win
  probability on average (v2's D2: 0.036).
- **Stage 2** (2 h, 28M positions): fine-tune weighted towards S2 puzzle positions (35%) and D3 (17%),
  lr 3e-4. Held-out got slightly *worse* (0.024 → 0.028), but every match improved, so stage 2 shipped.

**Result:** puzzles 59.3%, SF 1320 92%, SF 1600 73%, SF 2000 49% (performance 1,991). v3 beat v2 92% and
v1 95.5% head-to-head.

### 4.4 v3.1: ChessBench, and finding the capacity limit

ChessBench's action-value shards are shuffled across all 2,148 files, so a few shards cannot give
positions with every move graded. We used its **best-move file** instead (527M positions, Stockfish 16
at 50 ms), reading slices of the 36 GB file by HTTP range requests on Kaggle CPU notebooks, ~4k
positions/s, ~480 bytes per position.

Two 4-hour fine-tunes from v3 at equal GPU time: a control on our own data, and one with 35%
ChessBench (CB1). Both tied v3 (50.5% and 50.0% head-to-head) and neither beat it against SF 2000
(46.7%, 47.5%). Held-out improved a little (0.028 → 0.024). **Lesson: at 6.7M parameters v3 was
saturated.** More data, even fresh data, improved accuracy on positions but not play. Capacity was the
limit, so v4 went bigger.

### 4.5 v4: twice as deep

**Architecture.** 16 layers instead of 8 at the same width, so v3's weights seed it (§2.6), plus the
square-pair attention bias. 13.6M parameters.

**Pilots** (110 min each, grown from v3.1-cb): A = 16 layers + bias, B = 16 layers without bias.
Neither moved clearly (53% and 48.5% vs v3 over 100 games). Inspecting the weights showed why:

| | v3's layers 0–7 | new layers 8–15 after 2 h |
|---|---|---|
| norm of attention-output + feed-forward-output weights | 66–86 | **~3** |
| attention bias, mean absolute value | – | 0.011 |

The new layers started at zero and the fine-tuning learning rate (2e-4) was too small to grow them,
so the model was still essentially v3. **Fix:** train the new layers and the bias at 5× the learning
rate (1e-3, v3's from-scratch rate). In a CPU smoke test they then moved 5× as far per step.

**Main run** (10.5 h on 2×T4, from pilot A, 63M positions, ChessBench CB3 35% of the mix): the new
layers grew to norms of 16–37 and the bias to 0.18. Against v3, 200 games: 55.2% (EMA weights). An
unexplained ~15× slowdown hit the last ~3 hours (no error in the log); later runs did not repeat it.

**Where it loses.** `bench/blunders.py` graded every move v4 made in 120 games against SF 2000 at
depth 12, skipping the random opening plies:

| phase | moves | avg win prob lost | share of all loss | moves losing > 0.3 |
|---|---|---|---|---|
| opening (first 10 moves each) | 959 | 0.0175 | 12% | 1 |
| **middlegame** | 3,422 | **0.0289** | **70%** | **42** |
| endgame (≤ 13 points of material) | 1,769 | 0.0144 | 18% | 7 |

v3 showed the same split. The weak spot is middlegame tactics, not endgames, so tablebases were
deprioritised in favour of deeper labels on middlegame positions.

**Move-choice blend.** Tuning `mix` (§2.7) on held-out and puzzles favoured mix ≈ 2 (held-out 0.0229
vs 0.0232, puzzles 60.2% vs 59.1%). In games, against SF 2000 on the same 200 openings:

| | score | perf |
|---|---|---|
| v3 | 47.2% | 1,981 |
| v4 main EMA, mix 0 | 52.2% | 2,016 |
| v4 main EMA, mix 2 | 56.0% | 2,042 |

The blend was worth more against Stockfish (+4 points) than head-to-head (+1.7 points vs mix 0).

**Fine-tune** (3 h, from the main run's EMA and optimizer state, lr 1e-4): self-play D4 (4,000 games
from pilot A, 311k positions) 15%, fresh ChessBench CB4 30%, and unused Lichess shards. Puzzles rose
to 63.5%. This fine-tune's EMA **shipped as v4**: 66.5% vs v3 and 61.5% vs the main-run model
(200 games each), SF 2000 56.2% (perf 2,044), SF 2400 33.0%.

### 4.6 v4.1: deeper self-play labels

**D5:** 4,000 self-play games by the v4 main-run model, every legal move graded at **depth 12** instead
of 8. That takes ~2 h per 500 games per Kaggle CPU notebook (depth 8: 20 min), 311k positions.

**Fine-tune** (150 min from v4's EMA, lr 5e-5): D5 25%, D4 10%, ChessBench CB4 20%, every-move-graded
data ≥ 50%, 18.4M positions.

**Result**, 200 games each, mix 2: **56.8% vs v4** (+80 =67 −53, ~+47 Elo), **SF 2000 64.0%** (perf
2,100), SF 2400 33.0% (same as v4). Puzzles 64.3%. Shipped as v4.1.

### 4.7 v4.2: a round that did not help

**D6:** 4,000 games by v4.1 at depth 12, 320k positions. v4.1's own moves lost 0.0222 win probability
per move, against 0.0236 for the model that played D5: the bot is making smaller mistakes. A fine-tune
from v4.1 with D6 25% and D5 15% (40% self-play) **lost** to v4.1 46.0% head-to-head and was slightly
worse against SF 2000 (62% vs 64%). It first appeared to jump against SF 2400 (42% vs 33%), but a
second 200-game run gave 38.3% vs 39.3%: noise. Not shipped. A second self-play round on the same
recipe, at a 40% self-play share, did not add strength.

---

### 4.8 v1 retrained: the same size with everything we learned

To separate what came from scale from what came from data and method, we trained a fresh model at
**v1's size** (d = 128, 4 layers, 4 heads, 0.89M parameters) with the current encoding, heads, labels
and losses, from random weights. Training: 3 h on one T4 at lr 1e-3 with cosine decay, batch 1,024,
EMA, ~9,400 positions/s, **101M positions seen** (~250× v1's ~0.4M samples; drawn from ~41M distinct
positions, ~510× v1's 80k), from a mix of ChessBench CB3 (30%),
Lichess evals L/L2 (25%), every-move-graded puzzle positions S2 (20%) and all six self-play sets D1–D6
(22%), which carry the move history. No attention bias. `models/vibefish_v1_retrain.pt` / `_ema.pt`.

| | old v1 | **v1 retrained** | v2 | v3 |
|---|---|---|---|---|
| parameters | 0.88M | **0.89M** | 2.9M | 6.7M |
| positions seen in training | ~0.4M | **101M** | ~2M | ~67M |
| compute | 2 CPU cores, 1 h | **1×T4, 3 h** | 4 CPU cores, ~4.5 h | 2×T4, 5 h |
| puzzles solved | 10.5% | **58.4%** | 39.1% | 59.3% |
| held-out wp lost / top1 | 0.086 / 33% | **0.031 / 50.5%** | 0.054 / 42% | 0.028 / 52% |
| vs random (120) | +19 =99 −2 | **+113 =7 −0** | +106 =14 −0 | +118 =2 −0 |
| vs SF 1320 / 1600 / 2000 (120) | 12% / – / – | **87.5% / 67.9% / 35.0%** | 52% / 26% / 12% | 92% / 73% / 49% |
| head-to-head, 200 games | | **93.0% vs old v1, 89.0% vs v2, 38.5% vs v3** | | |

The EMA weights are used; `mix = 2` gave 38.3% (perf 1,917) against SF 2000 over 120 games, within noise
of policy-only, and cost a point on puzzles. **The same 0.89M network that drew 99 of 120 games against
a random mover now beats the 3.3× larger v2 89% and plays at about v3-stage-1 strength.** Data, labels
and training method account for most of the distance from v1 to v3; parameters account for the rest
(the retrained v1 is ~80 Elo below v3). Its held-out loss was still falling at the end of the run.

## 5. Results summary

All numbers are one forward pass per move. v4 and v4.1 use `mix = 2`.

| | v1 | v2 | v3 | v4 | **v4.1** |
|---|---|---|---|---|---|
| parameters | 0.88M | 2.9M | 6.7M | 13.6M | **13.6M** |
| positions seen in training | ~0.4M | ~2M | ~67M | ~100M (+ v3's) | **+18M** |
| compute | 2 CPU cores, 1 h | 4 CPU cores, ~4.5 h | 2×T4, 5 h | 2×T4, ~16 h | **2×T4, 2.5 h** |
| puzzles solved (1,050) | 10.5% | 39.1% | 59.3% | 63.5% | **64.3%** |
| held-out wp lost / move | 0.086 | 0.054 | 0.028 | 0.023 | **0.023** |
| vs random mover (120) | +19 =99 −2 | +106 =14 −0 | +118 =2 −0 | +117 =3 −0 | **+117 =3 −0** |
| vs SF 1320 (120) | 12% | 52% | 92% | 97% | 95% |
| vs SF 1600 (120) | – | 26% | 73% | 80% | **85%** |
| vs SF 2000 (200) | – | 12% (120) | 47% | 56% | **64%** |
| vs SF 2400 (200) | – | – | – | 33% | 33% |
| performance vs SF 2000 | – | ~1,650 | 1,981 | 2,044 | **2,100** |
| head-to-head vs previous version | – | 82% vs v1 | 92% vs v2 | 66.5% vs v3 | **56.8% vs v4** |

Training ran on ~36 Kaggle GPU-hours in total for v3–v4.2, plus free CPU notebooks for all Stockfish
grading and ChessBench conversion.

---

## 6. What we learned

1. **Labels are everything.** VibeFish cannot be stronger than its teacher. Every big step came
   from more or better Stockfish labels: every-move grades, self-play positions, deeper grading.
2. **Every input feature must be exercised by training data.** Untrained history embeddings turned a
   25%-puzzle model into one that lost 0–20 to SF 1320.
3. **Mate distance belongs in the labels.** `1 + 0.2/n` lets the bot finish games with no outside help.
4. **Self-play is the most valuable data per position**, but each round has to prove itself in
   matches. Two of the six self-play rounds (v2's round 2, v4.2's D6) made the model worse.
5. **Hand-made features pay off when compute is small:** attack counts, cheapest attacker, pins.
6. **A saturated model doesn't improve with more of the same.** v3.1 gained held-out accuracy and no
   games. The fix was capacity (v4), not data.
7. **Grown layers need a higher learning rate.** Identity-initialised layers barely moved at a
   fine-tuning learning rate; 5× fixed it.
8. **The action-value head is worth using at play time** once it is trained on enough every-move data
   (it lost to policy-only in v2, won in v4).
9. **Games decide, and they are noisy.** Held-out loss misled us more than once; 200-game Stockfish
   matches swing by ±6 points.
10. **Data and method beat size at this scale.** Retrained with everything above, v1's 0.89M network
    beat the 2.9M v2 89% and came within ~80 Elo of the 6.7M v3 (§4.8).
11. **Free compute is enough to get to ~2,000 on Stockfish's scale.** fp16 on both T4s, shard-wise
    data loading, CPU notebooks for grading, and long jobs split into resumable chunks.

---

## 7. Limitations

- **Our Elo is on Stockfish's limited-strength scale at 0.1 s per move**, not a human rating. A Lichess
  bot account would give a human-comparable number.
- **Tactics are still the weak spot:** 47% of puzzles rated 2000, 27% at 2400, 4% at 2800+, and 70% of
  the lost win probability against SF 2000 is in the middlegame.
- **Labels are shallow.** Most every-move grades are Stockfish at depth 8; only D5/D6 are at depth 12.
- **Training volume is ~1,000× smaller than DeepMind's.** v4's whole line has seen ~220M positions; their
  9M model saw ~41 billion.
- **One unexplained slowdown** in the v4 main run was never diagnosed.

## 8. Future work

- Self-play from v4.1 with a lower self-play share, or new kinds of data, rather than more of the same.
- Deeper regrading (depth 12–16) of the middlegame positions where the model loses the most.
- **HL-Gauss binned value and action-value heads** (DeepMind's best loss).
- **Mirror averaging** for positions without castling rights (two passes of the same network on the
  same position: still searchless).
- DeepMind's 10k test puzzles, for a direct comparison with their 9M and 270M models.
- int8 quantisation of the shipped model (~4× smaller).

---

## Appendix A: files and reproduction

| file | role |
|---|---|
| `model.py` | encoding, `ChessGPT`, `grow()`, move choice |
| `teacher.py` | Stockfish grading of every legal move, score → grade, self-play worker |
| `data.py` | Lichess, puzzle and ChessBench conversion, self-play jobs, `.npz` shards |
| `train.py` | trainer: mixed sources, cosine schedule, fp16, EMA, optimizer state, learning-rate boost |
| `evaluate.py` | puzzles, held-out, matches vs Stockfish and head-to-head (`--vs path@mix`) |
| `bench/blunders.py` | lost win probability by game phase from match games |
| `kaggle_run.py` | runs commands on Kaggle GPU/CPU notebooks with datasets attached |
| `play.py` | terminal play and UCI mode (defaults: v4.1, `--mix 2`) |
| `models/` | every trained model, shipped or not |
| `bench/results.jsonl`, `bench/train_logs/`, `games/` | every evaluation, training log and match game |
| `PLAN.md` | the full lab notebook: every run, command and number |

Exact commands, data mixes and Kaggle notebook names for each run are in `PLAN.md` and `HANDOFF.md`.

## Appendix B: credits

Trained in part on **ChessBench** (DeepMind, CC-BY 4.0). Every-move grades from **Stockfish 17.1**
(v1 used Stockfish 11). Positions from the **Lichess** evaluation and puzzle databases. Compute from
Kaggle's free tier.

## References

- A. Ruoss, G. Delétang, S. Medapati, et al. *Grandmaster-Level Chess Without Search.* arXiv:2402.04494, 2024.
- S. Ross, G. Gordon, D. Bagnell. *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning* (DAgger). AISTATS 2011.
- E. Imani, M. White. *Improving Regression Performance with Distributional Losses* (HL-Gauss). ICML 2018.
- Stockfish, https://stockfishchess.org. Lichess open database, https://database.lichess.org.

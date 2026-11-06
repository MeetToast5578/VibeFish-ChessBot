# Handoff (2026-11-06, ~20:05 UTC)

State of the work when it was paused, and what to do next. Read `CLAUDE.md` and `PLAN.md` too.

## Where things stand

- **Shipped:** `models/vibefish_v3.pt` (6.7M params, d=256, 8 layers). v1 is in `models/v1/`
  (`evaluate.py --v1 models/v1`, `--vs v1:models/v1`).
- **v3.1 experiment** (two 4 h fine-tunes from v3, lr 2e-4, both on Kaggle 2×T4):
  - `vibefish-v31-control` (our data only): **done, not shipped.** Even with v3: 50.5% head-to-head
    (+36 =29 −35), SF 2000 46.7% (v3 48.7%), puzzles 58.8% (v3 59.3%), held-out wp_lost 0.0252 /
    top1 52.9% (v3 0.028 / 52.2%). Lesson: more training on the same data no longer helps v3.
  - `vibefish-v31-cb` (35% ChessBench best-move data, CB1 = positions 0–20M): **was still training**,
    pushed 16:36 UTC, 240 min, so done ~20:45 UTC. Not benchmarked yet.
  - ChessBench shards on Kaggle: `vibefish-cb-1` (CB1_00–19, 20M) and `vibefish-cb-2` (positions
    20–40M) are both complete. cb-2 is unused so far. (`kaggle kernels output` lists sizes as
    "822"; the real files are ~484 MB each, per the kernel log.)
- **Evaluation check (in progress):** Stockfish's `UCI_Elo` is calibrated at 60s+0.6s, but our
  matches gave it 0.1 s/move, so our perf Elo numbers may be inflated. `evaluate.py --movetime`
  exists now (default 0.1). A run of v3 vs SF 2000 at **1.0 s/move, 60 games** was started locally
  at ~19:50 UTC (writes `games/v3_vs_sf2000_1s.pgn` and a line in `bench/results.jsonl` when done).
  If the session died, rerun it (~15 min):
  ```bash
  python evaluate.py --model models/vibefish_v3.pt --match 2000 --games 60 --movetime 1.0 --pgn games/v3_vs_sf2000_1s.pgn --log bench/results.jsonl
  ```
  Compare with v3's 48.7% at 0.1 s. A big drop means the Elo figures in README/PLAN need redoing
  at the longer time control. The owner's "v3 plays a lot of h4" worry was checked: early h4/...h5
  appears in 19% of v3's games vs 21–30% for Stockfish 1320–2000 in the same matches, so not a bug.

## Next steps, in order

1. **Benchmark the ChessBench run** (same as the control, ~5 min locally):
   ```bash
   kaggle kernels output yunus0901/vibefish-v31-cb -p kaggle_out/vibefish-v31-cb --file-pattern '(runs/|\.log$)'
   ```
   ```bash
   python evaluate.py --model kaggle_out/vibefish-v31-cb/runs/v31_cb/model.pt --puzzles --heldout --vs models/vibefish_v3.pt --games 100 --pgn games/v31_cb_vs_v3.pgn --log bench/results.jsonl
   ```
   ```bash
   python evaluate.py --model kaggle_out/vibefish-v31-cb/runs/v31_cb/model.pt --match 2000 --games 120 --pgn games/v31_cb_sf.pgn --log bench/results.jsonl
   ```
   Ship as v3.1 only if it clearly beats v3 head-to-head (≥ ~57% over 100 games). If it ships:
   copy to `models/vibefish_v3_1.pt`, update the `play.py`/`evaluate.py` defaults, README and the
   credit line "Trained in part on ChessBench (DeepMind, CC-BY 4.0)". Either way, record the result
   in PLAN.md (v3.1 section) and copy its `log.jsonl` to `bench/train_logs/v31_cb.jsonl`.
2. ~~Time-control check~~ done: v3 52.5% vs SF 2000 at 1 s/move (48.7% at 0.1 s). Elo figures stand.
3. **v4** (owner's direction: bigger and deeper, ≤ 14M; v3 isn't strong enough).

## v4 plan (agreed in chat; model code done, pilots not started)

Owner's rules: **≤ 14M params** (raised from 10M for v4: "bigger but not too big"), one forward
pass per move, no search; DeepMind models only as opponents, ChessBench data allowed with credit.

- **Architecture** (same width d=256 so v3's weights seed it): **16 layers** instead of 8, plus
  the **square-pair attention bias** (`[layers, heads, 68, 68]`, PLAN item 6). 13.6M params with
  the bias, ~27 MB at fp16. **Loops dropped:** with params allowed, 2 passes through N layers cost
  as much as 2N real layers with half the weights. The code still supports `--loops`.
- **Code (done, PR #18):** `ChessGPT(loops=, bias=)`, `model.grow`, `train.py --init ... --grow
  --layers 16 --attn-bias`. Added layers start as identity, so grown-from-v3 plays exactly like v3
  (tested). The bias is cast to the query dtype for fp16 autocast. Untested on GPU so far.
- **Pilots (~2 h each, in parallel on Kaggle, both grown from v3):** A = 16 layers + bias,
  B = 16 layers, no bias. Compare held-out and head-to-head vs v3. Same mix as the v3.1 winner.
- **Main run:** continue the pilot winner (`--init` its model.pt, no `--grow`) for 8–10 h. Expect
  ~2× slower per position than v3 (~1,800 pos/s on 2×T4, ~65M positions per 10 h). Use
  ChessBench (CB1 + cb-2's CB2) in the mix if the v3.1 CB run helped.
- **Budget:** 30 GPU h/week on Kaggle; the v3.1 runs used ~8 h this week.
- **Gate:** ship v4 only if it beats v3 (or v3.1) head-to-head and against Stockfish.

## Gotchas

- Activate the venv: `. /home/user/venv/bin/activate`.
- Stockfish here: `export STOCKFISH=/home/user/sf/stockfish/stockfish-ubuntu-x86-64-avx2`
  (without it, matches fail; puzzles/held-out/head-to-head don't need it).
- Kaggle: token is injected by the proxy, never stored or printed. Running kernels hide logs;
  poll `kaggle kernels status yunus0901/<name>`. `--file-pattern` avoids downloading big shards.
- Don't run Stockfish/data jobs next to a local training run.
- Every new feature must appear in training data (history features only come from self-play).

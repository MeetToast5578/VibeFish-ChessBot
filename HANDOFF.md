# Handoff (2026-11-06, ~20:05 UTC)

State of the work when it was paused, and what to do next. Read `CLAUDE.md` and `PLAN.md` too.

## Where things stand

- **Shipped:** `models/vibefish_v3.pt` (6.7M params, d=256, 8 layers). v1 is in `models/v1/`
  (`evaluate.py --v1 models/v1`, `--vs v1:models/v1`).
- **v3.1 experiment** (two 4 h fine-tunes from v3, lr 2e-4, both on Kaggle 2×T4):
  - `vibefish-v31-control` (our data only): **done, not shipped.** Even with v3: 50.5% head-to-head
    (+36 =29 −35), SF 2000 46.7% (v3 48.7%), puzzles 58.8% (v3 59.3%), held-out wp_lost 0.0252 /
    top1 52.9% (v3 0.028 / 52.2%). Lesson: more training on the same data no longer helps v3.
  - `vibefish-v31-cb` (35% ChessBench best-move data, CB1 = positions 0–20M): **done, not
    shipped.** Even with v3 too: 50.0% head-to-head (+31 =38 −31), SF 2000 52.5%, puzzles 59.1%,
    held-out wp_lost 0.0243 / top1 54.3% (best of the three). So v3's size is the limit, not data.
    Full table in PLAN.md (v3.1).
  - ChessBench shards on Kaggle: `vibefish-cb-1` (CB1_00–19, 20M) and `vibefish-cb-2` (positions
    20–40M) are both complete. cb-2 is unused so far. (`kaggle kernels output` lists sizes as
    "822"; the real files are ~484 MB each, per the kernel log.)
- **Evaluation check (in progress):** Stockfish's `UCI_Elo` is calibrated at 60s+0.6s, but our
  matches gave it 0.1 s/move, so our perf Elo numbers might have been inflated. Checked with
  `evaluate.py --movetime 1.0`: v3 vs SF 2000 at 1 s/move scored **50.0%** (+22 =16 −22, 60 games,
  `games/v3_vs_sf2000_1s.pgn`) vs 48.7% at 0.1 s. Stockfish's limited strength barely changes with
  more time, so the 0.1 s numbers stand (still on Stockfish's scale, not Lichess/FIDE). The owner's "v3 plays a lot of h4" worry was checked: early h4/...h5
  appears in 19% of v3's games vs 21–30% for Stockfish 1320–2000 in the same matches, so not a bug.

## Next steps, in order

1. **v4** (owner's direction: bigger, deeper, loop through the layers; v3 isn't strong enough).
   v3.1 is closed (see above). The benchmark recipe for any candidate:
   ```bash
   python evaluate.py --model <model.pt> --puzzles --heldout --vs models/vibefish_v3.pt --games 100 --pgn games/<name>_vs_v3.pgn --log bench/results.jsonl
   ```
   ```bash
   python evaluate.py --model <model.pt> --match 2000 --games 120 --pgn games/<name>_sf.pgn --log bench/results.jsonl
   ```
   Ship only if it clearly beats v3 head-to-head (≥ ~57% over 100 games). If ChessBench data is in
   a shipped model, add the README credit line "Trained in part on ChessBench (DeepMind, CC-BY 4.0)".

## v4 plan (agreed in chat, not started)

Owner's rules still hold: ≤ 10M params, one forward pass per move, no search; DeepMind models only
as opponents, ChessBench data allowed with credit.

- **Architecture** (same width d=256 so v3's weights can seed it):
  - **11 layers** instead of 8: ~9.4M params incl. the bias below (d=320 would exceed 10M).
  - **Loops:** run the layer stack 2× with shared weights, plus a learned per-pass offset
    (`loop_emb`). Effective depth 22, no extra params. Still one forward pass.
  - **Square-pair attention bias** (PLAN item 6, a light "smolgen"): a learned
    `[layers, heads, 68, 68]` additive bias (~0.4M params), passed as `src_mask` per layer.
    Needs `nn.TransformerEncoder`'s layers called one by one. Check it works under CUDA fp16
    autocast (float mask vs fp16 query) before a long run.
  - A first draft of these `model.py` changes was written and then set aside, untested. Keep
    `cfg` backwards compatible (new keys with defaults) so v1–v3 checkpoints still load.
- **Initialise from v3 ("depth up-scaling"):** copy v3's 8 layers, add 3 layers with `out_proj`
  and `linear2` zeroed (identity under pre-norm), zero bias, zero `loop_emb`. Needs a `train.py`
  path that loads a checkpoint into a bigger config (`strict=False` on matching keys). Note the
  second loop pass is not an identity at init; training has to adapt.
- **Test before the long run:** two ~2 h pilots in parallel on Kaggle, both from grown v3:
  A = 11 layers + bias, 1 loop; B = same with 2 loops. Compare held-out and head-to-head vs v3.
  Looping is the unproven part (helps in some research, not others); more params reliably help.
- **Main run:** the pilot winner, 8–10 h. Expect ~2.75× slower per position than v3 with 2 loops
  (~20M positions per 4 h instead of 53M), so the data fits in the 29 GB of RAM; memmap loading
  isn't needed yet. Use ChessBench (CB1 + cb-2's CB2) in the mix if the v3.1 CB run helped.
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

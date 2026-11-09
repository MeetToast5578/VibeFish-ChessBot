# Handoff (2026-11-08, ~15:45 UTC)

State of the work when it was paused, and what to do next. Read `CLAUDE.md` and `PLAN.md` too.

## Where things stand

- **Shipped: v4.1** = `models/vibefish_v4_1.pt` (copy of `vibefish_v41_ft_ema.pt`), played with
  `--mix 2` (`play.py`'s default). It is v4 (`models/vibefish_v4.pt`, PR #24) fine-tuned for 150 min
  (`vibefish-v41-ft`, lr 5e-5, from v4's EMA) with D5 25%, D4 10%, CB4 20%, every-move data ≥ 50%;
  18.4M positions, no slowdown. Every model of the line is kept in `models/`.
- **v4.1's numbers** (mix 2, 200 games unless noted): **56.8% vs v4** (+80 =67 −53), **SF 2000 64.0%**
  (perf 2100; v4 56.2%), SF 2400 33.0% (2277, same as v4), puzzles 64.3% (v4 63.5%), held-out
  0.0227. 120 games: SF 1600 84.6%, SF 1320 95%, random +117 =3 −0.
- **D5** (`vibefish-d5-1` / `-2` outputs, `D5_10`–`D5_23`): 4,000 games from the v4 main EMA,
  311k positions graded at **depth 12**, ~2 h per 500 games on a Kaggle CPU notebook (depth 8: 20 min).
  Its own moves lose 0.0236 wp on average at depth 12.
- **Where it loses** (v4, `bench/blunders.py` vs SF 2000): middlegame 70% of the lost win prob.
  Rerun it on `games/v41_ema_sf2000_2400.pgn` to see whether D5 moved that.

- **v1.1** (2026-11-08, `models/vibefish_v1_1_ema.pt`): v1's 0.89M size with the
  current recipe, 3 h on one T4, 101M positions. Puzzles 58.4%, 93% vs old v1, 89% vs v2, 38.5% vs v3,
  SF 2000 35%. Not a ship candidate (v4.1 is far stronger); it is the "same size, better method" study
  in PAPER.md §4.8. Its loss was still falling: a longer run or a self-play round would add more.

- **v2.1** (2026-11-09, `models/vibefish_v2_1_ema.pt`): v2's size (2.9M) with exactly v1.1's recipe and
  3 h budget; 41.9M positions (~3,900 pos/s). Puzzles 57.9%, held-out 0.028, **52.5% vs v1.1**, 89.2% vs
  old v2, 43.2% vs v3, SF 2000 42.1% (perf 1,945). First VibeFish X point: 3.3× params at equal GPU time
  ≈ +20–50 Elo. PAPER.md §4.9.
- **DeepMind opponents** (2026-11-09): `evaluate.py --vs "uci:sh bench/dm.sh 9M"` plays DeepMind's
  searchless models through their own code (`bench/dm_uci.py`, venv from `bench/dm_requirements.txt`,
  env DM_REPO / DM_CKPT / DM_PY; locally /home/user/google-deepmind, /home/user/dm_ckpt, /home/user/dmvenv).
  Their engine runs one forward pass per legal move (action value), ours one per position.
  **v4.1 vs DM 9M: +1 =55 −144 (14.2%), ~−310 Elo.** All losses are mates (median 88 plies), all
  draws repetitions. v4.1 vs 136M / 270M run on Kaggle CPU notebooks `vibefish-dm136m-a/b`,
  `vibefish-dm270m-a/b` (100 games each half, `--first-game 0/100`; outputs `results.jsonl` + PGN).
  vs 9M, 200 games: v1.1 13.0%, v2.1 12.2%, v3 9.5%. Most draws (34–42 per model) are DeepMind
  ≥3 points up and repeating instead of converting, so the real gap is bigger. PAPER.md §4.10.
  **DM 136M vs v4.1: +0 =26 −174 (6.5%).** 270M still running (`vibefish-dm270m-a/b`, backups `-c/-d`
  for games 0–99; games 100–199 have no backup yet: Kaggle allows 5 CPU sessions).

## Next steps, in order

0. **D6 done** (2026-11-08): 4,000 games from v4.1, 320k positions at depth 12, own moves lose
   0.0222 wp (D5 from v4-main: 0.0236). **v4.2 fine-tune running:** `vibefish-v42-ft`, launched
   12:31 UTC, 150 min from the `vibefish-v41-ft` output's `ema.pt`, lr 5e-5, D6 25% / D5 15% / D4 5% /
   CB4 15% / L, L2_1*, S2, D3, S_puz, D1, D2. **Result: not shipped.** 46.0% vs v4.1 head to head,
   SF 2000 62.0% (v4.1 64.0%), but SF 2400 42.0% (v4.1 33.0%). Models kept as
   `models/vibefish_v42_ft.pt` / `_ema.pt`. The SF 2400 jump was noise: a rerun gave v4.2 38.3%, v4.1
   39.3% (400 games: 40.1% vs 36.1%, ~1σ). SF 200-game runs swing ~6 points; use 400 for close calls.
1. **Self-play D6 from v4.1: started 02:08 UTC** (`vibefish-d6-1` / `-2`, seeds 160000–171500,
   puzzle rows from 4.8M, chunks `D6_10`–`D6_23`, done ~10:45 UTC). It was set up at depth 12 (same command as D5 with `--model` = the
   `vibefish-v41-ft` output's `runs/v41_ft/ema.pt`, new seeds and puzzle rows, e.g. seeds 150000+,
   `--start` 4,700,000+). Two CPU notebooks, ~8.5 h each. Then a fine-tune like `vibefish-v41-ft`
   (init from its `ema.pt`, D6 25%, D5 15%). Each round has been worth +45–80 Elo head to head.
2. Gate as before: 200 games vs `models/vibefish_v4_1.pt@2`, 200 vs SF 2000 / 2400, all `--mix 2`.
3. PLAN v4 items: mirror averaging (D14), HL-Gauss value heads (C12). Tablebases only if endgames
   start to matter.

GPU budget: v4 used ~18 GPU h and v4.1 ~2.7 h. Check the remaining weekly quota on Kaggle before a
long run; the reset day hasn't been confirmed.

## Gotchas

- Activate the venv: `. /home/user/venv/bin/activate`.
- Stockfish here: `export STOCKFISH=/home/user/sf/stockfish/stockfish-ubuntu-x86-64-avx2`
  (without it, matches fail; puzzles/held-out/head-to-head don't need it).
- Kaggle: token is injected by the proxy, never stored or printed. Running kernels hide logs;
  poll `kaggle kernels status yunus0901/<name>`. `--file-pattern` avoids downloading big shards.
- Don't run Stockfish/data jobs next to a local training run.
- Every new feature must appear in training data (history features only come from self-play).

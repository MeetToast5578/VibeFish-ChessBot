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

## Next steps, in order

0. **D6 done** (2026-11-08): 4,000 games from v4.1, 320k positions at depth 12, own moves lose
   0.0222 wp (D5 from v4-main: 0.0236). **v4.2 fine-tune running:** `vibefish-v42-ft`, launched
   12:31 UTC, 150 min from the `vibefish-v41-ft` output's `ema.pt`, lr 5e-5, D6 25% / D5 15% / D4 5% /
   CB4 15% / L, L2_1*, S2, D3, S_puz, D1, D2. **Result: not shipped.** 46.0% vs v4.1 head to head,
   SF 2000 62.0% (v4.1 64.0%), but SF 2400 42.0% (v4.1 33.0%). Models kept as
   `models/vibefish_v42_ft.pt` / `_ema.pt`. Next: rerun SF 2400 (200 more games each for v4.1 and
   v4.2-EMA, different openings via evaluate.py's game numbering) to see if the jump is real.
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

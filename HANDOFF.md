# Handoff (2026-11-07, ~15:05 UTC)

State of the work when it was paused, and what to do next. Read `CLAUDE.md` and `PLAN.md` too.

## Where things stand

- **Shipped: v4** = `models/vibefish_v4.pt` (copy of `vibefish_v4_ft_ema.pt`; 13.6M params, d=256,
  16 layers + square-pair attention bias), played with `--mix 2` (`play.py`'s default). Every step
  of v4 is kept in `models/` (pilots A/B, main, main EMA, fine-tune, fine-tune EMA). v3 and v1 stay too.
  Merged to main in PR #24 (#21–#23 brought the code: EMA/opt state, lr boost, `blunders.py`, `--mix`).
- **v4's numbers** (mix 2; README table, `bench/results.jsonl`): puzzles 63.5% (v3 59.3%), held-out
  0.0231 / 54.7%, vs v3 66.5% (200 games), vs v4-main 61.5%, SF 2000 56.2% (perf 2044; v3 47.2% on
  the same 200 openings), SF 2400 33% (2277), SF 1600 80%, SF 1320 97%, random +117 =3 −0.
- **Where it loses** (`bench/blunders.py`, vs SF 2000): middlegame 70% of the lost win prob, 42 of
  50 big blunders. The endgame is not the weak spot.
- **Running:** self-play **D5** (`vibefish-d5-1` / `-2`, Kaggle CPU): 4,000 games from the v4 main
  EMA, graded at **depth 12**, 500-game chunks `D5_10`–`D5_23`. Started 2026-11-07 ~10:45 UTC and
  still RUNNING at 15:00 (a depth-8 chunk took 20 min; depth 12 is much slower). Kaggle stops a
  session at 12 h (~22:45 UTC); finished chunks are kept either way. A check is scheduled for 17:00 UTC.
- **Unexplained:** the main run slowed ~15× after 7.3 h with no error. The 3 h fine-tune had no
  slowdown. Long runs should log often (`--eval-every 2000`).

## Next steps, in order

1. When D5 is done, check its log (`model's own moves lost ...`, chunk times). Then run a 2–3 h
   fine-tune from `models/vibefish_v4.pt` (no `--grow`; `--boost-from 8 --boost 2` or none, lr ~5e-5)
   with D5 ~25%, D4 10%, every-move data ≥ 50%. Mirror `vibefish-v4-ft`'s command (PLAN v4 C11 /
   README). Its `opt.pt` is in the `vibefish-v4-ft` output if continuing that optimizer.
   Launch sketch (the shipped weights are on Kaggle as the `vibefish-v4-ft` output's `ema.pt`):
   ```bash
   R=/kaggle/input/notebooks/yunus0901/vibefish-v4-ft/runs/v4_ft
   MIX='data/D5_*.npz:0.25,data/D4_*.npz:0.1,data/CB4_*.npz:0.2,data/L_*.npz:0.1:8,data/L2_1*.npz:0.1:8,data/S2_*.npz:0.15,data/D3_*.npz:0.05,data/S_puz.npz:0.02,data/D1.npz:0.015,data/D2.npz:0.015'
   python kaggle_run.py --name vibefish-v41-ft \
     --dataset yunus0901/vibefish-v2-data --dataset yunus0901/vibefish-v3-data --dataset yunus0901/vibefish-v3-d3 --dataset yunus0901/vibefish-v3-s2local \
     --kernel-source yunus0901/vibefish-grade-1 --kernel-source yunus0901/vibefish-v4-ft --kernel-source yunus0901/vibefish-cb-4 \
     --kernel-source yunus0901/vibefish-d4-1 --kernel-source yunus0901/vibefish-d4-2 --kernel-source yunus0901/vibefish-d5-1 --kernel-source yunus0901/vibefish-d5-2 \
     --cmd "python train.py --init $R/ema.pt --mix '$MIX' --minutes 150 --batch 1024 --lr 5e-5 --warmup 300 --eval-every 2000 --multi-gpu --out /kaggle/working/runs/v41_ft"
   ```
   Then download `runs/v41_ft` (`--file-pattern '(runs/v41_ft/(model|ema)\.pt|log\.jsonl|\.log$)'`),
   copy to `models/vibefish_v41_ft.pt` / `_ema.pt` and the log to `bench/train_logs/`.
2. Gate: 200 games vs `models/vibefish_v4.pt@2` (both at `--mix 2`) plus 200 vs SF 2000 / 2400.
   Ship as v4.1 only if it wins head-to-head and isn't worse against Stockfish.
3. Benchmark commands need `--mix 2` (evaluate.py's default is 0); opponents take `path@2`.
4. Then PLAN v4 items: mirror averaging (D14), HL-Gauss value heads (C12), tablebases only if
   endgames start to matter.

GPU budget: v4 used ~18 GPU h (pilots 4, main 11, fine-tune 3) and the v3.1 runs ~8 h. Check the
remaining weekly quota on Kaggle before a long run; the reset day hasn't been confirmed.

## Gotchas

- Activate the venv: `. /home/user/venv/bin/activate`.
- Stockfish here: `export STOCKFISH=/home/user/sf/stockfish/stockfish-ubuntu-x86-64-avx2`
  (without it, matches fail; puzzles/held-out/head-to-head don't need it).
- Kaggle: token is injected by the proxy, never stored or printed. Running kernels hide logs;
  poll `kaggle kernels status yunus0901/<name>`. `--file-pattern` avoids downloading big shards.
- Don't run Stockfish/data jobs next to a local training run.
- Every new feature must appear in training data (history features only come from self-play).

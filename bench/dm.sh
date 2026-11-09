#!/bin/sh
# Launch a DeepMind searchless_chess model as a UCI engine: sh bench/dm.sh 9M|136M|270M
# Needs: DM_REPO = parent dir of the searchless_chess checkout, DM_CKPT = dir holding checkpoints/<model>,
# DM_PY = python of a venv with jax, dm-haiku, orbax-checkpoint, chess. One CPU thread per engine.
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DM_CKPT/work" && cd "$DM_CKPT/work" || exit 1   # their loader reads ../checkpoints/<model>
export PYTHONPATH="$DM_REPO" XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" JAX_PLATFORMS=cpu
exec "$DM_PY" "$here/dm_uci.py" "$1"

"""Run VibeFish commands on a Kaggle GPU. Packs the code into a script notebook, pushes it, and
(with --wait) downloads the outputs when it finishes. Needs the Kaggle CLI and a token.

  python kaggle_run.py --name vibefish-smoke --dataset yunus0901/vibefish-v2-data \
      --cmd "python train.py --mix 'data/S_0.npz:1' --minutes 5 --out /kaggle/working/runs/smoke" --wait

Inside the notebook the files of all attached datasets are linked into ./data, and anything written to /kaggle/working is
kept as output (put --out there). The code runs from /tmp so the data link isn't saved as output.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import tarfile
import tempfile
import time

CODE = ["model.py", "teacher.py", "data.py", "train.py", "evaluate.py", "bench/exclude.txt", "bench/heldout_fens.txt",
        "bench/dm_uci.py", "bench/dm.sh", "bench/dm_requirements.txt"]

JOB = '''import base64, glob, io, os, subprocess, sys, tarfile
tarfile.open(fileobj=io.BytesIO(base64.b64decode("{code}"))).extractall("/tmp/vf")
os.chdir("/tmp/vf")
subprocess.run("pip install -q chess zstandard", shell=True, check=True)
os.makedirs("data")   # every attached dataset's files, side by side
for f in glob.glob("/kaggle/input/**/*.np[zy]", recursive=True) + glob.glob("/kaggle/input/**/*.pkl", recursive=True):
    os.symlink(f, "data/" + os.path.basename(f))
assert os.path.exists("data/heldout.npz"), "dataset not attached"
os.makedirs("/kaggle/working/runs", exist_ok=True)
subprocess.run("nvidia-smi || true", shell=True)
for cmd in {cmds!r}:
    print(">>>", cmd, flush=True)
    if subprocess.run(cmd, shell=True).returncode:
        sys.exit(1)
'''


def pack():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in CODE:
            tar.add(f)
    return base64.b64encode(buf.getvalue()).decode()


def kaggle(*args):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="notebook slug, e.g. vibefish-smoke")
    p.add_argument("--user", default="yunus0901")
    p.add_argument("--dataset", action="append", default=[])
    p.add_argument("--cmd", action="append", required=True)
    p.add_argument("--gpu", default="NvidiaTeslaT4")
    p.add_argument("--cpu", action="store_true", help="CPU-only notebook (doesn't use GPU quota), e.g. for Stockfish grading")
    p.add_argument("--tpu", action="store_true", help="TPU notebook (Kaggle TpuV5E8, separate 20 h/week quota)")
    p.add_argument("--kernel-source", action="append", default=[], help="attach another notebook's output, user/slug")
    p.add_argument("--wait", action="store_true", help="poll until done, then download outputs to kaggle_out/<name>")
    a = p.parse_args()
    ref = f"{a.user}/{a.name}"
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/job.py", "w").write(JOB.format(code=pack(), cmds=a.cmd))
        json.dump({"id": ref, "title": a.name, "code_file": "job.py", "language": "python",
                   "kernel_type": "script", "is_private": True, "enable_gpu": not (a.cpu or a.tpu), "enable_tpu": a.tpu,
                   "enable_internet": True, "machine_shape": "TpuV5E8" if a.tpu else None if a.cpu else a.gpu, "dataset_sources": a.dataset,
                   "kernel_sources": a.kernel_source}, open(f"{d}/kernel-metadata.json", "w"))
        r = kaggle("kernels", "push", "-p", d)
        print(r.stdout.strip(), r.stderr.strip())
        if r.returncode:
            raise SystemExit(1)
    if not a.wait:
        return
    while True:
        time.sleep(60)
        s = kaggle("kernels", "status", ref).stdout.strip()
        print(time.strftime("%H:%M"), s, flush=True)
        if any(k in s.lower() for k in ("complete", "error", "cancel")):
            break
    out = f"kaggle_out/{a.name}"
    os.makedirs(out, exist_ok=True)
    r = kaggle("kernels", "output", ref, "-p", out)
    print(r.stdout.strip()[-2000:], r.stderr.strip()[-2000:])


if __name__ == "__main__":
    main()

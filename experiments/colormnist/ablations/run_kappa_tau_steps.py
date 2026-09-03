"""Run the kappa / tau / num-steps ablation sweep for latent-space NDTM (colorMNIST).

One-at-a-time sweeps around the tuned baseline
(kappa=3e-4, tau=0, steps=200) at fixed gamma=5 on the in-distribution latent
DDPM, 3 seeds each, 10000 test queries per run. Serial (the 2080 Ti is
compute-bound during guidance, so concurrency gives no speedup).

Resumable: each finished run drops a marker in DONE_DIR; existing markers are
skipped. Big sample .pt files go to SAMPLES_DIR under $FFF_OUTPUT_ROOT, not into
the repository. Progress is appended to PROGRESS.

  python -m experiments.colormnist.ablations.run_kappa_tau_steps
  python -m experiments.colormnist.ablations.run_kappa_tau_steps --dry-run
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from experiments.common import paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The sampler is run as a module so it resolves its imports the same way a
# direct invocation does; the checkpoint is train_latent_diffusion.py's default.
SAMPLER = ["-m", "experiments.colormnist.sample_ndtm"]
INDIST_CKPT = paths.output("colormnist", "latent_diffusion_cc_mnist",
                           "checkpoints", "last.pt", create=False)

ROOT = paths.output("ablation_kappa_tau_steps", create=False)
SAMPLES_DIR = os.path.join(ROOT, "samples")
DONE_DIR = os.path.join(ROOT, "done")
LOG_DIR = os.path.join(ROOT, "logs")
PROGRESS = os.path.join(ROOT, "progress.txt")

GAMMA = 5.0
LIMIT = 10000
BATCH = 500
SEEDS = [0, 1, 2]

# baseline (shared point of all three sweeps)
BASE_KAPPA, BASE_TAU, BASE_STEPS = 3.0e-4, "zero", 200

# sweep grids (baseline value excluded here; the baseline runs cover it)
# Grids finalized from a 500-query coarse probe (fiber loss at gamma=5):
#   kappa flat ~0.057-0.061 for 0..1e-2, then 0.17 at 1e-1  -> trace the knee 1e-2..3e-1
#   tau   flat ~0.058 for 0..1e-2, 0.093 at 1, 1.02 at 100  -> trace the knee 1e-2..100
#   steps fiber 0.336/0.089/0.059/0.037 at 25/100/200/400 -> still improving at 400,
#         so extend to 800 (near the 1000-step training resolution) for the tail
#
# Note on the steps axis: 400 and 800 do not divide 1000, so they only run the
# number of steps they name because get_timesteps spreads a concentrated grid
# rather than returning a short one. 25, 50, 100 and the 200 baseline divide
# 1000 and are exact either way.
KAPPAS = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]               # w_control (7)
TAUS = ["1e-4", "1e-2", "1e-1", "1", "10", "100"]                # w_score (6)
STEPS = [25, 50, 100, 400, 800]                                  # num diffusion steps (5)


def build_runs():
    runs = []
    for s in SEEDS:
        runs.append(dict(tag=f"base_seed{s}", kappa=BASE_KAPPA, tau=BASE_TAU, steps=BASE_STEPS, seed=s))
    for k in KAPPAS:
        for s in SEEDS:
            runs.append(dict(tag=f"kappa_{k:g}_seed{s}", kappa=k, tau=BASE_TAU, steps=BASE_STEPS, seed=s))
    for t in TAUS:
        for s in SEEDS:
            runs.append(dict(tag=f"tau_{t}_seed{s}", kappa=BASE_KAPPA, tau=t, steps=BASE_STEPS, seed=s))
    for st in STEPS:
        for s in SEEDS:
            runs.append(dict(tag=f"steps_{st}_seed{s}", kappa=BASE_KAPPA, tau=BASE_TAU, steps=st, seed=s))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for d in (SAMPLES_DIR, DONE_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)

    runs = build_runs()
    pending = [r for r in runs if not os.path.exists(os.path.join(DONE_DIR, r["tag"]))]
    print(f"{len(runs)} runs total, {len(pending)} pending, {len(runs)-len(pending)} already done")
    if args.dry_run:
        for r in runs:
            mark = "DONE" if os.path.exists(os.path.join(DONE_DIR, r["tag"])) else "pending"
            print(f"  [{mark}] {r['tag']:24s} kappa={r['kappa']:<8g} tau={str(r['tau']):<6} steps={r['steps']}")
        return

    env = dict(os.environ)
    for i, r in enumerate(pending):
        t0 = datetime.now()
        cmd = [
            sys.executable, *SAMPLER,
            "--diffusion-ckpt", INDIST_CKPT,
            "--gamma", str(GAMMA), "--limit", str(LIMIT), "--batch-size", str(BATCH),
            "--w-control", f"{r['kappa']:g}", "--w-score", str(r["tau"]),
            "--num-steps", str(r["steps"]), "--seed", str(r["seed"]),
            "--output-dir", SAMPLES_DIR, "--tag", r["tag"],
        ]
        log_path = os.path.join(LOG_DIR, r["tag"] + ".log")
        print(f"[{i+1}/{len(pending)}] {t0:%H:%M:%S} running {r['tag']} "
              f"(kappa={r['kappa']:g} tau={r['tau']} steps={r['steps']}) ...", flush=True)
        with open(log_path, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
        dt = (datetime.now() - t0).total_seconds()
        # extract final fiber loss for the progress log
        fl = "?"
        try:
            with open(log_path) as lf:
                for line in lf:
                    if "TOTAL fiber loss: mean" in line:
                        fl = line.split("mean")[1].split("|")[0].strip()
        except Exception:
            pass
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        line = (f"{datetime.now():%H:%M:%S} {status} {r['tag']:24s} "
                f"kappa={r['kappa']:<8g} tau={str(r['tau']):<6} steps={r['steps']:<4} "
                f"fiber={fl} {dt:.0f}s")
        print("   " + line, flush=True)
        with open(PROGRESS, "a") as pf:
            pf.write(line + "\n")
        if rc == 0:
            open(os.path.join(DONE_DIR, r["tag"]), "w").close()
        else:
            print(f"   !! run failed; see {log_path}. Stopping so the failure is visible.", flush=True)
            sys.exit(rc)

    print("all pending runs complete", flush=True)


if __name__ == "__main__":
    main()

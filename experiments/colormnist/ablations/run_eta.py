"""Eta (DDIM stochasticity) ablation for latent-space NDTM (colorMNIST).

Follow-up to the kappa/tau/num-steps ablation. The reverse process in
fff/ndtm.py is a generalized DDIM update
  x_s = sqrt(alpha_s)*x0_hat + c2*eps + c1*z,   c1 = sqrt(1 - alpha_t/alpha_s)*eta
With variance_type="small" (the tuned config), eta interpolates the full
deterministic-DDIM -> DDPM-ancestral continuum:
  eta=0   -> deterministic DDIM (no injected noise, c1=0)
  eta=1   -> DDPM ancestral sampling (c1 = posterior std sqrt(beta~_t))
The tuned baseline is eta=0.25 (lightly stochastic DDIM).

Sweep eta in {0, 0.25, 0.5, 0.75, 1.0} at the tuned baseline
(kappa=3e-4, tau=0, steps=200, gamma=5) on the in-distribution latent DDPM,
10000 test queries x 3 seeds. eta=0.25 reuses the original ablation baseline
files (base_seed{s}) instead of rerunning.

Resumable via done-markers, same pattern as run_steps_gamma_matched.py.

  python -m experiments.colormnist.ablations.run_eta
  python -m experiments.colormnist.ablations.run_eta --dry-run
"""

import argparse
import glob
import os
import shutil
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

ROOT = paths.output("ablation_eta", create=False)
SAMPLES_DIR = os.path.join(ROOT, "samples")
DONE_DIR = os.path.join(ROOT, "done")
LOG_DIR = os.path.join(ROOT, "logs")
PROGRESS = os.path.join(ROOT, "progress.txt")

# reuse the original sweep's baseline files for the eta=0.25 point instead of rerunning
ORIG_SAMPLES_DIR = paths.output("ablation_kappa_tau_steps", "samples", create=False)

GAMMA = 5.0
LIMIT = 10000
BATCH = 500
SEEDS = [0, 1, 2]

BASE_KAPPA, BASE_TAU, BASE_STEPS = 3.0e-4, "zero", 200
BASE_ETA = 0.25

ETAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def build_runs():
    runs = []
    for e in ETAS:
        for s in SEEDS:
            runs.append(dict(tag=f"eta_{e:g}_seed{s}", eta=e, seed=s,
                             reuse_base=(abs(e - BASE_ETA) < 1e-12)))
    return runs


def try_reuse_baseline(run):
    """eta=0.25 is identical to the original sweep's base_seed{s}; copy it in."""
    src_tag = f"base_seed{run['seed']}"
    matches = glob.glob(os.path.join(
        ORIG_SAMPLES_DIR,
        f"sampled_colormnist_latent_space_invariances_gamma=5.0_{src_tag}_*.pt"))
    if len(matches) != 1:
        return False
    dst = os.path.join(SAMPLES_DIR, os.path.basename(matches[0]).replace(src_tag, run["tag"]))
    shutil.copy2(matches[0], dst)
    return True


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
            reuse = " (reuse baseline)" if r["reuse_base"] else ""
            print(f"  [{mark}] {r['tag']:20s} eta={r['eta']:<5g}{reuse}")
        return

    env = dict(os.environ)
    for i, r in enumerate(pending):
        t0 = datetime.now()
        if r["reuse_base"] and try_reuse_baseline(r):
            print(f"[{i+1}/{len(pending)}] {r['tag']} (reused from original sweep baseline)", flush=True)
            line = f"{datetime.now():%H:%M:%S} ok(reused) {r['tag']:20s} eta={r['eta']:<5g} (reused baseline)"
            with open(PROGRESS, "a") as pf:
                pf.write(line + "\n")
            open(os.path.join(DONE_DIR, r["tag"]), "w").close()
            continue

        cmd = [
            sys.executable, *SAMPLER,
            "--diffusion-ckpt", INDIST_CKPT,
            "--gamma", f"{GAMMA:g}", "--limit", str(LIMIT), "--batch-size", str(BATCH),
            "--w-control", f"{BASE_KAPPA:g}", "--w-score", str(BASE_TAU),
            "--num-steps", str(BASE_STEPS), "--eta", f"{r['eta']:g}", "--seed", str(r["seed"]),
            "--output-dir", SAMPLES_DIR, "--tag", r["tag"],
        ]
        log_path = os.path.join(LOG_DIR, r["tag"] + ".log")
        print(f"[{i+1}/{len(pending)}] {t0:%H:%M:%S} running {r['tag']} (eta={r['eta']:g}) ...", flush=True)
        with open(log_path, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
        dt = (datetime.now() - t0).total_seconds()
        fl = "?"
        try:
            with open(log_path) as lf:
                for line in lf:
                    if "TOTAL fiber loss: mean" in line:
                        fl = line.split("mean")[1].split("|")[0].strip()
        except Exception:
            pass
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        line = f"{datetime.now():%H:%M:%S} {status} {r['tag']:20s} eta={r['eta']:<5g} fiber={fl} {dt:.0f}s"
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

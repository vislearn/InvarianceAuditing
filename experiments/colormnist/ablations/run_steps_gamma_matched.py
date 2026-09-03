"""Number-of-diffusion-steps ablation with guidance strength (gamma) matched to steps.

Follow-up to the kappa/tau/num-steps ablation (run_kappa_tau_steps.py): the const
gamma-schedule only guides timesteps t<500, so the number of *guided correction
steps* scales with --num-steps (halving steps ~halves guided corrections at a
fixed gamma). Here gamma is scaled inversely with steps to hold the guidance
"dose" roughly constant: gamma = BASE_GAMMA * (BASE_STEPS / steps), e.g. 100
steps (half of the 200 baseline) gets gamma=10 (double).

kappa/tau stay at the tuned baseline (3e-4 / 0). The steps=200 point is
identical to the original ablation's baseline (gamma=5*(200/200)=5), so its
3 seed files are copied over instead of rerun.

The dose matching only works if --num-steps N actually runs N steps, which
requires N to divide the 1000 diffusion timesteps: 25, 50, 100 and 200 do.

Resumable via done-markers, same pattern as run_kappa_tau_steps.py.

  python -m experiments.colormnist.ablations.run_steps_gamma_matched
  python -m experiments.colormnist.ablations.run_steps_gamma_matched --dry-run
"""

import argparse
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

ROOT = paths.output("ablation_steps_gamma_matched", create=False)
SAMPLES_DIR = os.path.join(ROOT, "samples")
DONE_DIR = os.path.join(ROOT, "done")
LOG_DIR = os.path.join(ROOT, "logs")
PROGRESS = os.path.join(ROOT, "progress.txt")

# reuse the original sweep's baseline files for the steps=200 point instead of rerunning
ORIG_SAMPLES_DIR = paths.output("ablation_kappa_tau_steps", "samples", create=False)

LIMIT = 10000
BATCH = 500
SEEDS = [0, 1, 2]

BASE_KAPPA, BASE_TAU = 3.0e-4, "zero"
BASE_STEPS, BASE_GAMMA = 200, 5.0

STEPS = [25, 50, 100, 200, 400, 800]


def gamma_for(steps):
    return BASE_GAMMA * (BASE_STEPS / steps)


def build_runs():
    runs = []
    for st in STEPS:
        g = gamma_for(st)
        for s in SEEDS:
            runs.append(dict(tag=f"steps_{st}_gamma_{g:g}_seed{s}", steps=st, gamma=g, seed=s,
                              reuse_base=(st == BASE_STEPS)))
    return runs


def try_reuse_baseline(run):
    """steps=200/gamma=5 is identical to the original sweep's base_seed{s}; copy it in."""
    src_tag = f"base_seed{run['seed']}"
    import glob
    matches = glob.glob(os.path.join(ORIG_SAMPLES_DIR, f"sampled_colormnist_latent_space_invariances_gamma=5.0_{src_tag}_*.pt"))
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
            print(f"  [{mark}] {r['tag']:28s} steps={r['steps']:<4} gamma={r['gamma']:<6g}{reuse}")
        return

    env = dict(os.environ)
    for i, r in enumerate(pending):
        t0 = datetime.now()
        if r["reuse_base"] and try_reuse_baseline(r):
            dt = 0.0
            fl_note = "(reused from original sweep baseline)"
            print(f"[{i+1}/{len(pending)}] {r['tag']} {fl_note}", flush=True)
            line = f"{datetime.now():%H:%M:%S} ok(reused) {r['tag']:28s} steps={r['steps']:<4} gamma={r['gamma']:<6g} {fl_note}"
            with open(PROGRESS, "a") as pf:
                pf.write(line + "\n")
            open(os.path.join(DONE_DIR, r["tag"]), "w").close()
            continue

        cmd = [
            sys.executable, *SAMPLER,
            "--diffusion-ckpt", INDIST_CKPT,
            "--gamma", f"{r['gamma']:g}", "--limit", str(LIMIT), "--batch-size", str(BATCH),
            "--w-control", f"{BASE_KAPPA:g}", "--w-score", str(BASE_TAU),
            "--num-steps", str(r["steps"]), "--seed", str(r["seed"]),
            "--output-dir", SAMPLES_DIR, "--tag", r["tag"],
        ]
        log_path = os.path.join(LOG_DIR, r["tag"] + ".log")
        print(f"[{i+1}/{len(pending)}] {t0:%H:%M:%S} running {r['tag']} "
              f"(steps={r['steps']} gamma={r['gamma']:g}) ...", flush=True)
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
        line = (f"{datetime.now():%H:%M:%S} {status} {r['tag']:28s} "
                f"steps={r['steps']:<4} gamma={r['gamma']:<6g} fiber={fl} {dt:.0f}s")
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

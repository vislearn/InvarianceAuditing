import h5py
import numpy as np
from scipy.special import kl_div, rel_entr
import torch
import lightning_trainable
import os
import fff
import matplotlib.pyplot as plt
from fff.evaluate.fid import compute_fid_fast, reference_statistics
import argparse
import re


device = "cuda" if torch.cuda.is_available() else "cpu"

from fff.utils.checkpoint import trusted_load
from experiments.common import paths

# Trained conditional fiber models, one directory per run (see configs/colormnist).
log_folder = os.environ.get("FIBER_MODEL_LOGS", "lightning_logs")
plot_folder = paths.output("colormnist", create=False)

def Decolorize(x_colored):
    def detect_colors(x_data):
        background_colors = torch.mean(x_data[:,:,:,0],-1)
        return background_colors
    x_c = x_colored.reshape(-1,3,28,28)
    c = detect_colors(x_c)
    c_image = c.unsqueeze(-1).expand(-1,3,28*28).reshape(-1,3,28,28)
    x_dc = (x_c-c_image) / ((c_image+0.5)%1 - c_image)
    return x_dc.abs(), c

def normal(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi)/sigma
def gaussian_mix_dense(x):
    return 0.6 * normal(x, 0.7, 0.08) + 0.35 * normal(x, 0.5, 0.015) + 0.05 * normal(x, 0.1, 0.02)


class SubjectModelInterface:
    def __init__(self, subject_model):
        self.subject_model = subject_model

    def __call__(self, x):
        return self.subject_model.encode(x, torch.empty((x.shape[0], 1), device=x.device))

SEED_SUFFIX = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)$")


def is_known_run(name, known):
    """True for a configured run, or a seed repeat of one.

    A run trained with a non-default seed is named `<config stem>_seed<N>`.
    Those are the same config and belong in the same table, so they are scored
    like any other run rather than rejected -- which is what makes a seed sweep
    measurable.
    """
    if name in known:
        return True
    m = SEED_SUFFIX.match(name)
    return bool(m) and m.group("base") in known


def load_model(name):
    """Load a trained conditional fiber model by run name.

    Hyperparameters recorded by older revisions are brought up to the current
    schema by FiberModelHParams._migrate_hparams, so nothing needs patching here
    beyond pointing the recorded paths at this machine.
    """
    root = os.path.join(log_folder, name)
    if not os.path.isdir(root):
        # The bare failure named a relative 'lightning_logs/<run>', which is the
        # default only because 38_/39_ export FIBER_MODEL_LOGS and a hand-run
        # command does not. Say where we looked and what is actually there.
        available = sorted(os.listdir(log_folder)) if os.path.isdir(log_folder) else None
        hint = (f"\n{log_folder} does not exist either -- set FIBER_MODEL_LOGS to the "
                "training log directory, e.g.\n"
                "  export FIBER_MODEL_LOGS=$FFF_OUTPUT_ROOT/colormnist/lightning_logs"
                if available is None else
                f"\nruns present in {log_folder}: "
                + (", ".join(available) if available else "(none)"))
        raise SystemExit(f"no training logs for run {name!r} at {root}{hint}")
    try:
        checkpoint = lightning_trainable.utils.find_checkpoint(
            root=root, version=0, epoch="best")
    except Exception:
        checkpoint = lightning_trainable.utils.find_checkpoint(
            root=root, version=0, epoch="last")

    with trusted_load():
        ckpt = torch.load(checkpoint, map_location=device)
    hparams = ckpt["hyper_parameters"]
    hparams["data_set"]["root"] = paths.data("cc_mnist")
    hparams["data_set"]["subject_model_path"] = paths.data("cc_mnist", "subject_model.ckpt")
    hparams["load_lossless_ae_path"] = paths.data("cc_mnist", "lossless_vae.ckpt")

    model = fff.FiberModel(hparams)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval().to(device)


def save_model_samples(samples, originals, sample_embeddings, original_embeddings, name):
    save_path = os.path.join(plot_folder, name)
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        "samples": samples,
        "originals": originals,
        "sample_embeddings": sample_embeddings,
        "original_embeddings": original_embeddings
    }, os.path.join(save_path, "samples.pt"))

def save_model_stats(fl_stats, kl_stats, w1_stats, dev_stats, fid_stats, name):
    save_path = os.path.join(plot_folder, name)
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        "fl_stats": fl_stats,
        "kl_stats": kl_stats,
        "w1_stats": w1_stats,
        "dev_stats": dev_stats,
        "fid_stats": fid_stats,
    }, os.path.join(save_path, "stats.pt"))
    
def load_model_samples(name):
    load_path = os.path.join(plot_folder, name, "samples.pt")
    samples_dict = torch.load(load_path)
    return samples_dict["samples"], samples_dict["originals"], samples_dict["sample_embeddings"], samples_dict["original_embeddings"], 


def load_model_stats(name):
    load_path = os.path.join(plot_folder, name, "stats.pt")
    stats_dict = torch.load(load_path, weights_only=False)
    return stats_dict["fl_stats"], stats_dict["kl_stats"], stats_dict["w1_stats"], stats_dict["dev_stats"], stats_dict["fid_stats"],

def cached_stats(name, skip_fid=False):
    """Previously computed stats for `name`, or None.

    FID is by far the most expensive number here and never changes for a fixed
    samples.pt, so a finished model is not scored twice and a rerun costs
    nothing. Stats older than their samples.pt are ignored: that is the one way
    they can be stale.
    """
    stats_file = os.path.join(plot_folder, name, "stats.pt")
    samples_file = os.path.join(plot_folder, name, "samples.pt")
    if not os.path.exists(stats_file):
        return None
    if os.path.exists(samples_file) and os.path.getmtime(stats_file) < os.path.getmtime(samples_file):
        print(f"{name}: stats.pt is older than samples.pt, recomputing")
        return None
    stats = load_model_stats(name)
    # A cached NaN FID means it was skipped, not computed: reuse it only when
    # this run is skipping FID too, otherwise fall through and fill it in.
    if not skip_fid and any(np.isnan(x) for x in stats[-1]):
        return None
    return stats


@torch.no_grad()
def evaluate_model(model_name, dataset=None, samples_per_image=10, batch_size=512, save=True,
                   fid_slots=None, skip_fid=False, recompute=False):
    print(f"Evaluating model {model_name}")
    if not recompute:
        stats = cached_stats(model_name, skip_fid=skip_fid)
        if stats is not None:
            print(f"{model_name}: stats already computed, skipping (--recompute to redo)")
            return stats
    fiber_model = load_model(model_name)
    subject_model = SubjectModelInterface(fiber_model.subject_model)
    
    if dataset is not None:
        if not isinstance(dataset, torch.utils.data.TensorDataset):
            dataset = torch.utils.data.TensorDataset(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    try:
        samples, originals, sample_embeddings, original_embeddings = load_model_samples(model_name)
        if dataset is not None:
            if torch.mean((dataset[:][0] - originals)**2) > 1.e-6:
                raise ValueError("Dataset not identical to precomputed dataset")
            else:
                print("Using precomputed dataset")
    except Exception as e:
        # print(e)
        assert dataset is not None, "If no precomputed samples are available, dataset has to be passed"
    
        samples = []
        originals = []
        sample_embeddings = []
        original_embeddings = []
        
        for n_batch, batch in enumerate(dataloader):
            x = batch[0].to(device).float()
            test_image_embedding = subject_model(x)

            samples_image = []
            embeddings_image = []
            for i in range(samples_per_image):
                samples_image.append(fiber_model.sample(torch.Size([x.shape[0]]), test_image_embedding.to(device)).reshape(x.shape[0], 3, 28, 28))
                embeddings_image.append(subject_model(samples_image[-1]))
            samples.append(torch.stack(samples_image, dim=1))
            sample_embeddings.append(torch.stack(embeddings_image, dim=1))
            originals.append(x)
            original_embeddings.append(test_image_embedding)
            
        samples = torch.cat(samples, dim=0)
        sample_embeddings = torch.cat(sample_embeddings, dim=0)
        originals = torch.cat(originals, dim=0)
        original_embeddings = torch.cat(original_embeddings, dim=0)
        
        if save:
            save_model_samples(samples, originals, sample_embeddings, original_embeddings, model_name)
    
    fl_stats = compute_fiber_loss_model(originals.to(device), 
                                        samples.to(device), 
                                        original_embeddings.to(device), 
                                        sample_embeddings.to(device), 
                                        subject_model)
    kl_stats, w1_stats, dev_stats = compute_kl_w1_and_deviation(samples.permute(1, 0, 2, 3, 4))
    if skip_fid:
        # NaN is make_figures' "not computed yet" sentinel: it fills the column in
        # later with `make_figures fid`, and the table prints an empty cell.
        print("Skipping FID (--skip_fid)")
        fid_stats = (float("nan"), float("nan"))
    else:
        fid_stats = compute_fid_with_std(originals.to(device), samples.to(device),
                                         slots=fid_slots)
    if save:
        save_model_stats(fl_stats, kl_stats, w1_stats, dev_stats, fid_stats, model_name)
    return fl_stats, kl_stats, w1_stats, dev_stats, fid_stats
    
@torch.no_grad()
def compute_fiber_loss_model(originals, samples, original_embeddings, sample_embeddings, subject_model, batch_size=2048):
    N=None
    n_rows=5
    fiber_loss = []
    
    dataset = torch.utils.data.TensorDataset(originals, samples, original_embeddings, sample_embeddings)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("Computing Fiber Loss...")
    
    for sam_i in range(samples.shape[1]):
        fiber_loss_sami = []

        for i, batch in enumerate(dataloader):
            test_samples, x_sampled, test_c, xc = batch
            x_sampled, xc = x_sampled[:,sam_i], xc[:,sam_i]
            if i == 0 and sam_i == 0:
                # sanity check
                xc_recomputed = subject_model(x_sampled)
                assert torch.sqrt(torch.mean((xc-xc_recomputed)**2,-1)).mean() < 1.e-4, f"Fiber loss between identical samples is {torch.sqrt(torch.sum((xc-xc_recomputed)**2,-1)/float(xc.shape[-1])).mean()}"
            

            fiber_loss_sami.append(torch.sqrt(torch.sum((xc-test_c)**2,-1)/float(xc.shape[-1])))
        fiber_loss.append(torch.cat(fiber_loss_sami,0).cpu())
    
    fl_mean, fl_std = torch.cat(fiber_loss, dim=0).mean(), torch.cat(fiber_loss, dim=0).std() 
    print("fiber_loss mean is: ", fl_mean, " +- ", fl_std)
    
    return (fl_mean, fl_std)

@torch.no_grad()
def compute_fid_with_std(originals, samples, slots=None):
    """FID per sample slot, against one reference embedded once.

    Every slot and every model scores against the same originals, so their
    Inception activations are computed on the first call and reused -- which is
    half the work here, and all of it after the first model.

    `slots` caps how many sample slots are scored. Each one is a full Inception
    pass over the test set, so this is the knob that decides the runtime; fewer
    slots means a noisier standard deviation, not a different mean.
    """
    n_slots = samples.shape[1] if slots is None else min(slots, samples.shape[1])
    print(f"Computing FID over {n_slots} of {samples.shape[1]} sample slots...")
    reference_stats = reference_statistics(originals)
    fids = []
    for sam_i in range(n_slots):
        fids.append(compute_fid_fast(originals, samples[:, sam_i],
                                     reference_stats=reference_stats))
    fid_mean, fid_std = np.mean(fids), np.std(fids)
    print("fid mean is: ", fid_mean, " +- ", fid_std)

    return (fid_mean, fid_std)

def compute_kl_w1_and_deviation(sample_list, n_bins=100):
    print("Computing KL Divergence...")
    kls_r, kls_g, kls_b = [], [], []
    w1s_r, w1s_g, w1s_b = [], [], []
    dev = []
    for samples in sample_list:
        x_dc, colors = Decolorize(samples)
        max_pix = torch.max(x_dc.mean(1).reshape(-1,28*28), -1)[0].cpu()
        dev.append(torch.abs(max_pix - 1).mean().numpy())

        kls_c, w1s_c = [], []
        for c in range(3):
            H, bins = np.histogram(colors[:, c].cpu(), bins=n_bins, range=[0,1], density=True)
            bin_width = bins[1] - bins[0]
            mids = bins[:-1] + bin_width / 2

            # Discretized probabilities
            p = H * bin_width
            q = gaussian_mix_dense(mids) * bin_width
            p /= p.sum()
            q /= q.sum()

            # KL divergence
            kl_per_bin = rel_entr(p, q)
            kl_per_bin = kl_per_bin[~np.logical_or(np.isnan(kl_per_bin), np.isinf(kl_per_bin))]
            kl = np.sum(kl_per_bin)
            kls_c.append(kl)

            # Wasserstein-1 distance
            C_p = np.cumsum(p)
            C_q = np.cumsum(q)
            tv_per_bin = np.abs(C_p - C_q)
            tv_per_bin = tv_per_bin[~np.logical_or(np.isnan(tv_per_bin), np.isinf(tv_per_bin))]
            w1 = np.sum(tv_per_bin) * bin_width
            w1s_c.append(w1)

        kls_r.append(kls_c[0]); kls_g.append(kls_c[1]); kls_b.append(kls_c[2])
        w1s_r.append(w1s_c[0]); w1s_g.append(w1s_c[1]); w1s_b.append(w1s_c[2])

    # Aggregate statistics
    def summarize(values):
        mean = np.mean(values)
        std = np.std(values)
        return mean, std

    kl_means = [summarize(x) for x in [kls_r, kls_g, kls_b]]
    w1_means = [summarize(x) for x in [w1s_r, w1s_g, w1s_b]]
    dev_mean, dev_std = summarize(dev)

    print("Red KL mean is: ", kl_means[0][0], " +- ", kl_means[0][1])
    print("Green KL mean is: ", kl_means[1][0], " +- ", kl_means[1][1])
    print("Blue KL mean is: ", kl_means[2][0], " +- ", kl_means[2][1])

    return kl_means, w1_means, (dev_mean, dev_std)


def _bool(value: str) -> bool:
    """argparse type for a genuine boolean flag.

    `type=bool` calls bool() on the string, so `--include_test_data False` is
    True and the flag cannot be turned off. Accepts the bare flag as well as an
    explicit value, so both spellings work.
    """
    if value.lower() in ("1", "true", "yes", "y"):
        return True
    if value.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=str, default="all",
                    help="Which model class to evaluate")
    parser.add_argument("--models", type=str, default=None,
                    help="Comma-separated run names to evaluate, e.g. "
                         "'fff_lambda1,nf_lambda10'. Overrides --model_class. Each "
                         "run writes to its own directory, so this shards cleanly "
                         "across jobs -- one model per array task is the finest split, "
                         "and drawing the samples plus FID for one model is the unit "
                         "of work worth parallelising.")
    parser.add_argument("--include_test_data", type=_bool, nargs="?", const=True,
                    default=False,
                    help="Whether to provide test data")
    parser.add_argument("--fid_slots", type=int, default=None,
                    help="Score only this many sample slots for FID (default: all 10). "
                         "Each slot is a full Inception pass over the 40k test set, so "
                         "this is the main runtime knob. Affects only the reported std.")
    parser.add_argument("--skip_fid", action="store_true",
                    help="Leave the FID column empty. Everything else -- fiber loss, "
                         "KL, W1, deviation -- is cheap and still computed. Fill FID in "
                         "later with `python -m experiments.colormnist.make_figures fid`.")
    parser.add_argument("--recompute", action="store_true",
                    help="Score models again even if their stats.pt is already there. "
                         "Without this, a finished model is skipped, which makes "
                         "rerunning the step cheap.")
    args = parser.parse_args()

    if not args.skip_fid and device == "cpu":
        print("WARNING: no GPU visible. FID runs Inception over 40k images per sample "
              "slot, which is tens of minutes each on CPU. Consider a GPU node, "
              "--fid_slots 3, or --skip_fid.", flush=True)

    # One entry per config in configs/colormnist/fiber_models; the key is the
    # --name the run was launched with, which is where load_model looks for it.
    model_names = {
        "fff": ["fff_lambda0", "fff_lambda1", "fff_lambda10", "fff_lambda100"],
        "fif": ["fif_lambda0", "fif_lambda1", "fif_lambda10", "fif_lambda100"],
        "nf": ["nf_lambda0", "nf_lambda1", "nf_lambda10", "nf_lambda100"],
        "dnf": ["dnf_lambda0", "dnf_lambda1", "dnf_lambda10", "dnf_lambda100"],
        "mlf": ["mlf_lambda0", "mlf_lambda1", "mlf_lambda10"],
        "diff": ["diff_lambda0"],
        "fm": ["fm_lambda0"],
    }

    if args.include_test_data:
        with h5py.File(paths.data("cc_mnist", "data.h5"), 'r') as f:
            test_data = torch.from_numpy(f['test_images'][:])
        print("Recomputing statistics with test data")
    else:
        test_data = None
        print("Using precomputed statistics")
        
    if args.models:
        known = {name for names in model_names.values() for name in names}
        selected = [n.strip() for n in args.models.split(",") if n.strip()]
        unknown = [n for n in selected if not is_known_run(n, known)]
        if unknown:
            raise SystemExit(f"unknown run name(s): {', '.join(unknown)}\n"
                             f"known: {', '.join(sorted(known))}\n"
                             f"(a '<run>_seed<N>' repeat of a known run is also accepted)")
    elif args.model_class == "all":
        selected = [name for names in model_names.values() for name in names]
    else:
        if args.model_class not in model_names:
            raise SystemExit(f"unknown model class {args.model_class!r}; "
                             f"known: {', '.join(model_names)}, or 'all'")
        selected = model_names[args.model_class]

    print(f"Evaluating {len(selected)} model(s): {', '.join(selected)}", flush=True)
    for model_name in selected:
        evaluate_model(model_name, dataset=test_data,
                       fid_slots=args.fid_slots, skip_fid=args.skip_fid,
                       recompute=args.recompute)

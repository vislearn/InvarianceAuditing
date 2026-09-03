"""The nearest-neighbour baseline of Table 5.

For every audited image, the closest *real* image in the dataset under the
subject model's own representation. It is the reference the invariant samples
are read against: a fiber sample whose loss is far below the nearest neighbour's
is closer to the query, in the model's eyes, than any genuine image is.

The samplers do not produce this. `experiments/*/nearest_neighbours.py` fill the
`nn_embeddings` key in afterwards, and the evaluators report the column as soon
as it is there.

Ported from `find_nearest_neighbor_batched_mm` in the evaluation notebooks,
including its two conventions worth knowing about:

* the search ranks by the *mean* squared difference over feature dimensions,
  while the loss finally reported is the *sum*. Both are minimised by the same
  neighbour, so the ranking is unaffected;
* `identity_threshold` drops candidates that are closer than 1e-4, which is how
  the query image excludes itself when it is a member of the search set.
"""

import torch
from tqdm import tqdm


@torch.no_grad()
def nearest_neighbour_indices(query_embeddings, dataset, subject_model,
                              identity_threshold: float = 1e-4,
                              batch_size: int = 256, metric: str = "l2",
                              progress: bool = True, num_workers: int = 0,
                              query_chunk: int = 4096):
    """Index into `dataset` of the nearest real image to each query embedding.

    `query_embeddings` is (N, D) and already on `subject_model`'s device. The
    dataset is streamed, so the candidate embeddings are never all held at once
    and N is free to be large -- which is the point: the callers pool every run
    of a setting into one call, so the dataset is embedded once per setting
    rather than once per run.

    The l1 and cross-entropy branches materialise a (B, N, D) tensor, which at a
    pooled N would be tens of gigabytes, so they walk the queries in blocks of
    `query_chunk`. The l2 branch needs no such care: it expands the square into
    one (B, N) matmul.
    """
    device = query_embeddings.device
    n_queries, dim = query_embeddings.shape
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        # Respawning workers per epoch costs more than the single pass they do.
        persistent_workers=False)

    best_dist = torch.full((n_queries,), torch.inf, device=device)
    best_idx = torch.full((n_queries,), -1, dtype=torch.long, device=device)

    if metric == "l2":
        query_norms = (query_embeddings ** 2).mean(dim=1)
    elif metric == "cross_entropy":
        query_probs = query_embeddings.softmax(dim=-1)
    elif metric != "l1":
        raise ValueError(f"unknown metric {metric!r}")

    offset = 0
    for batch in tqdm(loader, disable=not progress, desc="nearest neighbours"):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        candidates = subject_model(images.to(device, non_blocking=True))
        size = candidates.shape[0]

        if metric == "l2":
            # ||a - b||^2 / D expanded, so only one (B, N) matmul is materialised
            dist = ((candidates ** 2).mean(dim=1)[:, None]
                    + query_norms[None, :]
                    - 2 * (candidates @ query_embeddings.T) / dim)
        else:
            dist = torch.empty((size, n_queries), device=device)
            log_candidates = (candidates.log_softmax(dim=-1)
                              if metric == "cross_entropy" else None)
            for start in range(0, n_queries, query_chunk):
                stop = min(start + query_chunk, n_queries)
                if metric == "l1":
                    block = (candidates[:, None, :]
                             - query_embeddings[None, start:stop, :]).abs().mean(dim=-1)
                else:
                    block = -(query_probs[None, start:stop, :]
                              * log_candidates[:, None, :]).sum(dim=-1)
                dist[:, start:stop] = block

        # A candidate this close is the query itself; it is not its own neighbour.
        dist = torch.where(dist < identity_threshold,
                           torch.full_like(dist, torch.inf), dist)

        chunk_dist, chunk_idx = dist.min(dim=0)
        better = chunk_dist < best_dist
        best_dist = torch.where(better, chunk_dist, best_dist)
        best_idx = torch.where(better, chunk_idx + offset, best_idx)
        offset += size

    if (best_idx < 0).any():
        raise RuntimeError(
            "no nearest neighbour found for some queries -- every candidate was "
            "within identity_threshold, so the search set is probably the query "
            "set itself with duplicates")
    return best_idx


@torch.no_grad()
def nearest_neighbour_embeddings(query_embeddings, dataset, subject_model,
                                 also_embed=None, batch_size: int = 256,
                                 num_workers: int = 0, **kwargs):
    """-> (indices, embeddings, {name: cross embeddings}).

    `also_embed` maps a name to another subject model, which is applied to the
    same neighbour images -- the cross-model view Figures 8 and 18 compare.
    """
    device = query_embeddings.device
    indices = nearest_neighbour_indices(query_embeddings, dataset, subject_model,
                                        batch_size=batch_size,
                                        num_workers=num_workers, **kwargs)
    order = indices.tolist()

    def embed(model):
        # Fetched and embedded a batch at a time. Stacking every neighbour image
        # first is what the notebook did, and it is fine for one run's few
        # hundred queries -- but these calls are now pooled over a whole setting,
        # where ten thousand 256x256 images would be ~8 GB of RAM before a single
        # forward pass.
        out = []
        for start in range(0, len(order), batch_size):
            images = torch.stack(
                [dataset[i][0] if isinstance(dataset[i], (list, tuple)) else dataset[i]
                 for i in order[start:start + batch_size]], dim=0)
            out.append(model(images.to(device)).cpu())
        return torch.cat(out, dim=0)

    cross = {name: embed(model) for name, model in (also_embed or {}).items()}
    return indices.cpu(), embed(subject_model), cross

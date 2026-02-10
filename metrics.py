import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional import pairwise_cosine_similarity


def calculate_apd(embs_1: torch.Tensor, embs_2: torch.Tensor) -> float:
    embs_1 = torch.nn.functional.normalize(embs_1, dim=1)
    embs_2 = torch.nn.functional.normalize(embs_2, dim=1)
    # Cosine distance (1 - cosine similarity)
    pairwise_distances = 1 - pairwise_cosine_similarity(embs_1, embs_2)
    
    # Compute the sum of pairwise distances
    sum_distances = torch.sum(pairwise_distances)
    
    # Calculate the average pairwise distance (APD)
    apd = sum_distances / (len(embs_1) * len(embs_2)) # Lengths should be the same
    
    return apd.item()


def calculate_prt(embs_1: torch.Tensor, embs_2: torch.Tensor, eps: float = 1e-8) -> float:
    c1 = embs_1.mean(dim=0)
    c2 = embs_2.mean(dim=0)

    c1 = c1 / (c1.norm() + eps)
    c2 = c2 / (c2.norm() + eps)

    cos = torch.dot(c1, c2)

    return (1.0 / (cos + eps)).item()

def calculate_minimum_distance(embs_1: torch.Tensor, embs_2: torch.Tensor, corpus_order: int) -> float:
    if corpus_order == 1:
        min_distances = torch.min(torch.cdist(embs_1, embs_2), dim=1).values
    elif corpus_order == 2:
        min_distances = torch.min(torch.cdist(embs_2, embs_1), dim=1).values

    avg_min_distance = torch.mean(min_distances)
    return avg_min_distance.item()

def calculate_amd(embs_1: torch.Tensor, embs_2: torch.Tensor) -> float:
    # For each point in embs_1, find the minimum distance to any point in embs_2
    min_distances_1 =calculate_minimum_distance(embs_1, embs_2, corpus_order=1)
    
    # For each point in embs_2, find the minimum distance to any point in embs_1
    min_distances_2 = calculate_minimum_distance(embs_1, embs_2, corpus_order=2)
    
    # Calculate the average of these minimum distances
    return (min_distances_1 + min_distances_2) / 2

def calculate_greedy_symmetric_amd(
    embs_1: torch.Tensor,
    embs_2: torch.Tensor,
    reduce: str = "mean",
) -> float:
    """
    Symmetric AMD via greedy one-to-one matching:
      - compute all pairwise distances
      - repeatedly take the smallest remaining distance
      - remove the matched row+col (each point used at most once)
    Returns mean (or sum) of matched distances.

    Notes:
      - If |embs_1| != |embs_2|, matches only min(n1, n2) pairs.
      - Complexity: O(n1*n2) memory; greedy loop O(m * (n1*n2)) if you recompute argmin,
        but we do it by masking, still typically fine for small n (WiC/LSCD usually small).
    """
    assert embs_1.dim() == 2 and embs_2.dim() == 2, "Inputs must be [N, D] tensors"
    n1, d1 = embs_1.shape
    n2, d2 = embs_2.shape
    assert d1 == d2, "Embedding dims must match"

    if n1 == 0 or n2 == 0:
        return float("nan")

    # Pairwise distance matrix [n1, n2]
    dist = torch.cdist(embs_1, embs_2)

    # Greedy matching with masking
    used1 = torch.zeros(n1, dtype=torch.bool, device=dist.device)
    used2 = torch.zeros(n2, dtype=torch.bool, device=dist.device)

    matched = []
    num_pairs = min(n1, n2)

    for _ in range(num_pairs):
        # mask out used rows/cols by setting them to +inf
        masked = dist.clone()
        if used1.any():
            masked[used1, :] = float("inf")
        if used2.any():
            masked[:, used2] = float("inf")

        # find global min
        flat_idx = torch.argmin(masked)
        i = flat_idx // n2
        j = flat_idx % n2

        val = masked[i, j]
        if torch.isinf(val):
            break  # no valid pairs left (shouldn't happen unless NaNs/Infs)

        matched.append(val)
        used1[i] = True
        used2[j] = True

    if len(matched) == 0:
        return float("nan")

    matched = torch.stack(matched)
    if reduce == "mean":
        return matched.mean().item()
    elif reduce == "min":
        return matched.min().item()
    elif reduce == "max":
        return matched.max().item()
    else:
        raise ValueError("reduce must be 'mean', 'min', or 'max'")
    

def calculate_optimal_symmetric_amd(embs_1: torch.Tensor, embs_2: torch.Tensor) -> float:
    """
    Optimal one-to-one symmetric AMD using Hungarian assignment.
    Matches min(n1, n2) points by minimum total distance.
    Returns mean matched distance.
    """

    assert embs_1.dim() == 2 and embs_2.dim() == 2
    n1 = embs_1.size(0)
    n2 = embs_2.size(0)
    if n1 == 0 or n2 == 0:
        return float("nan")

    dist = torch.cdist(embs_1, embs_2).detach().cpu().numpy()  # [n1, n2]
    row_ind, col_ind = linear_sum_assignment(dist)  # size = min(n1, n2) if rectangular
    matched = dist[row_ind, col_ind]
    return float(np.mean(matched))


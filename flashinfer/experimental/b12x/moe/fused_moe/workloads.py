"""Deterministic, untimed routing inputs for MoE candidate comparisons."""

from __future__ import annotations

from collections import Counter
import random

import torch


TUNING_WORKLOAD_VERSION = "statistical_pair_shared_v1"
ROUTING_MIX = ("diffuse", "skewed", "correlated")
ROUTING_WORKLOADS = (
    *ROUTING_MIX,
    "pair_shared",
    "disjoint",
    *(f"shared_{p}" for p in (0, 20, 40, 60, 80, 100)),
)


def make_routing_ids(
    tokens: int,
    top_k: int,
    num_experts: int,
    *,
    workload: str = "shared_40",
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build routes with distinct experts per token.

    Statistical scenarios sample lognormal expert popularity; ``correlated``
    also shares random preferences within four-token groups. ``pair_shared``
    shares preferences and sampling noise within adjacent token pairs.
    ``shared_N`` reuses N% of the batch's token/expert assignments, rounded
    to the nearest realizable expert count. Reuse favors already popular
    experts, rather than giving every expert exactly one or two tokens.
    One-token batches cannot share; a small expert pool may force more reuse.
    ``disjoint`` retains the cyclic, maximally spread baseline.
    """
    if tokens < 1 or not 1 <= top_k <= num_experts:
        raise ValueError("require positive tokens and 1 <= top_k <= num_experts")
    if workload == "disjoint":
        return (
            torch.arange(tokens * top_k, device=device, dtype=torch.int32)
            .reshape(tokens, top_k)
            .remainder_(num_experts)
        )
    if workload not in ROUTING_WORKLOADS:
        raise ValueError(f"unknown routing workload: {workload!r}")
    if workload in (*ROUTING_MIX, "pair_shared"):
        return _make_statistical_routing_ids(
            tokens, top_k, num_experts, workload=workload, seed=seed, device=device
        )
    return _make_shared_routing_ids(
        tokens,
        top_k,
        num_experts,
        sharing_percent=int(workload.removeprefix("shared_")),
        seed=seed,
        device=device,
    )


def _make_shared_routing_ids(
    tokens: int,
    top_k: int,
    num_experts: int,
    *,
    sharing_percent: int,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    if tokens < 1 or not 1 <= top_k <= num_experts:
        raise ValueError("require positive tokens and 1 <= top_k <= num_experts")
    unique = min(
        num_experts, max(top_k, (tokens * top_k * (100 - sharing_percent) + 50) // 100)
    )
    rng = random.Random(seed)
    expert_ids = rng.sample(range(num_experts), unique)
    counts: Counter[int] = Counter()
    rows = []
    introduced = 0
    for token in range(tokens):
        target = top_k + (unique - top_k) * token // max(tokens - 1, 1)
        new_count = target - introduced
        row = expert_ids[introduced:target]
        candidates = list(counts)
        for _ in range(top_k - new_count):
            expert = rng.choices(
                candidates, weights=[counts[e] ** 2 for e in candidates], k=1
            )[0]
            row.append(expert)
            candidates.remove(expert)
        rng.shuffle(row)
        rows.append(row)
        counts.update(row)
        introduced = target
    return torch.tensor(rows, dtype=torch.int32, device=device)


def _make_statistical_routing_ids(
    tokens: int,
    top_k: int,
    num_experts: int,
    *,
    workload: str,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample distinct experts with persistent popularity and group correlation."""
    expert_std, group_std = {
        "diffuse": (0.5, 0.0),
        "skewed": (1.5, 0.0),
        "correlated": (0.8, 2.0),
        "pair_shared": (0.8, 2.0),
    }[workload]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    popularity = torch.randn(num_experts, generator=generator) * expert_std
    logits = popularity.expand(tokens, -1).clone()
    group_size = 2 if workload == "pair_shared" else 4
    group_count = (tokens + group_size - 1) // group_size
    if group_std:
        groups = torch.randn(group_count, num_experts, generator=generator)
        logits += groups.repeat_interleave(group_size, dim=0)[:tokens] * group_std
    if workload == "pair_shared":
        uniform = torch.rand(group_count, num_experts, generator=generator)
        uniform = uniform.repeat_interleave(group_size, dim=0)[:tokens].clamp_min_(1e-7)
    else:
        uniform = torch.rand(tokens, num_experts, generator=generator).clamp_min_(1e-7)
    logits -= (-uniform.log()).log()
    return logits.topk(top_k, dim=-1).indices.to(device=device, dtype=torch.int32)


def make_tuning_routes(
    tokens: int,
    top_k: int,
    num_experts: int,
    *,
    device: torch.device | str,
    seed: int = 104729,
    samples_per_workload: int = 2,
) -> torch.Tensor:
    """Cover independent routes, expert popularity, and local token correlation.

    Diffuse, skewed, and correlated scenarios vary expert popularity; correlated
    scenarios also share preferences within groups of four tokens. ``pair_shared``
    sends adjacent token pairs to one set of experts while allowing other pairs to
    choose differently. Whole-batch sharing remains available as a separate
    diagnostic workload; it is not assigned a fixed production frequency.
    Each scenario uses two seeds by default.
    These weights provide workload coverage, not measured traffic frequencies.
    """
    if samples_per_workload < 1:
        raise ValueError("samples_per_workload must be positive")
    return torch.stack(
        [
            make_routing_ids(
                tokens,
                top_k,
                num_experts,
                workload=workload,
                seed=seed + tokens + 1009 * index,
                device=device,
            )
            for workload in (*ROUTING_MIX, "pair_shared")
            for index in range(samples_per_workload)
        ]
    )

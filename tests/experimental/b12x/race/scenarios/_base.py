"""Shared types and helpers for race-test scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class Scenario:
    forward: Callable[[], None]
    outputs: dict[str, torch.Tensor]
    reset: Callable[[], None] | None = None
    close: Callable[[], None] = lambda: None
    keepalive: list = field(default_factory=list)
    description: str = ""


def _rows_for_waves(device, n, waves, tile_m=64, tile_n=128) -> int:
    """Rows that give a 64x128-tiled output about ``waves`` CTAs per SM. Races
    between CTAs need more CTAs than SMs, and their rate grows with the number
    of waves: on a 188-SM RTX PRO 6000, a dense GEMM without its stage-reuse
    proxy fence fails 0.04% of contended replays as a one-wave 96-row grid and
    1-2% as a several-wave 512-row grid."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    tiles_n = -(-n // tile_n)
    return tile_m * max(1, -(-waves * sms // tiles_n))


def _generator(case) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(case.get("seed", 5)))

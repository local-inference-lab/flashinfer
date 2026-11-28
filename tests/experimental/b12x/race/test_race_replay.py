"""b12x kernels must give sound results while other work perturbs their timing.

Each scenario (see ``scenarios/``) is captured in a CUDA graph and its first
replay becomes the reference. Quiet replays calibrate how far arithmetic order
alone moves each output (b12x is not deterministic, and need not be); every
replay under ``SMContention`` must then stay within a margin of that spread
(see ``replay.py``). Values outside it mean the kernel's result depends on CTA
or warp timing: a missing fence, barrier release or CTA-wide sync. A b12x
dense GEMM without its stage-reuse proxy fence (mutation
``dense_gemm_stage_fence``) puts about 1-4% of contended replays outside it on
GB10, RTX PRO 4000 and RTX PRO 6000.

Budgets and modes:

- ``FLASHINFER_B12X_RACE_REPLAYS`` (default 2000): contended replays per case.
- ``FLASHINFER_B12X_RACE_FUZZ`` (default 0): random extra cases per family,
  drawn with ``FLASHINFER_B12X_RACE_SEED`` (default 0).
- ``B12X_RACE_FUZZ_DELAY=1`` (set before pytest starts): compile every kernel
  with random warp delays after CTA syncs, named barriers and mbarrier waits
  (``b12x._lib.race_fuzz``), which exposes kernels that publish results before
  the rest of their CTA has written them. Instrumented kernels get their own
  compile cache entries. The two modes detect different defects, so run the
  suite in both: contention exposes missing proxy fences, which the delays
  hide (``dense_gemm_stage_fence`` fails 28-40 of 2000 contended replays on an
  RTX PRO 6000 without delays and none with them), and the delays expose
  missing CTA syncs before barrier arrivals.

A curated case may carry ``"xfail": reason`` for a known, unfixed race; it
runs as a strict xfail, so the suite reports when a fix makes it pass.

``python -m tests.experimental.b12x.race.mutations`` puts fixed races back
into b12x and checks that these scenarios catch them.
"""

from __future__ import annotations

import os

import pytest

from ..conftest import require_b12x
from . import scenarios
from .noise import SMContention
from .replay import ReplayCheck

REPLAYS = int(os.environ.get("FLASHINFER_B12X_RACE_REPLAYS", "2000"))
FUZZ = int(os.environ.get("FLASHINFER_B12X_RACE_FUZZ", "0"))
SEED = int(os.environ.get("FLASHINFER_B12X_RACE_SEED", "0"))


@pytest.fixture(scope="module")
def contention():
    return SMContention(require_b12x())


def _check(family, case, contention):
    device = require_b12x()
    scenario = scenarios.FAMILIES[family](device, case)
    try:
        check = ReplayCheck(scenario.forward, scenario.outputs, reset=scenario.reset)
        with contention:
            report = check.run(replays=REPLAYS, contention=contention)
    finally:
        scenario.close()
    assert report.noise_kernels, "no noise kernels ran"
    assert report.bad == 0, f"{scenario.description} ({case}): {report}"


@pytest.mark.parametrize(
    "family,case",
    [
        pytest.param(
            family,
            case,
            id=f"{family}-{index}",
            marks=pytest.mark.xfail(strict=True, reason=case["xfail"])
            if "xfail" in case
            else (),
        )
        for family, cases in scenarios.CASES.items()
        for index, case in enumerate(cases)
    ],
)
def test_replays_match_under_contention(family, case, contention):
    _check(family, case, contention)


@pytest.mark.parametrize(
    "family,case",
    [
        pytest.param(family, case, id=f"{family}-fuzz{index}")
        for family in scenarios.FAMILIES
        for index, case in enumerate(scenarios.fuzz_cases(family, FUZZ, SEED))
    ]
    or [
        pytest.param(
            None, None, marks=pytest.mark.skip(reason="FLASHINFER_B12X_RACE_FUZZ=0")
        )
    ],
)
def test_fuzzed_replays_match_under_contention(family, case, contention):
    _check(family, case, contention)

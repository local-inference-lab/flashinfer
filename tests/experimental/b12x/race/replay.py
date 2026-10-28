"""CUDA-graph replay checks for b12x kernels under perturbation.

A scenario supplies ``forward`` (launches the kernels under test on the current
stream from fixed inputs), the ``outputs`` it writes, and optionally ``reset``
(restores anything ``forward`` mutates, such as accumulators or state; it is
recorded in the same graph ahead of ``forward``). ``ReplayCheck`` captures
both and takes the first replay as the reference.

b12x does not promise bit-exact results: atomics, split-K and scheduling may
reorder arithmetic from one replay to the next, and every order is fine. A race
instead produces values that no arithmetic order explains: a stale or
half-written tile, rows read before they were published. So the check measures
each floating-point output's error as ``max |x - ref|`` over the output's scale
(``max |ref|``), calibrates the spread that quiet replays show, and flags a
replay whose error exceeds both ``margin`` times that spread and ``floor`` (a
fraction of the scale); a quiet spread above ``broken`` means the scenario's
output is not meaningful to check. A non-finite value where the reference is
finite, or the reverse, always counts. Integer outputs must match exactly; a
scenario whose integer results legitimately vary (such as tie order in a
top-k) must canonicalize them or expose the values they select instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch


class NoisyScenario(AssertionError):
    """Quiet replays disagree beyond what the check can tolerate."""


@dataclass
class RaceReport:
    replays: int
    bad: int
    worst: dict[str, float]
    limits: dict[str, float]
    first: list[dict] = field(default_factory=list)
    noise_kernels: int | None = None

    def __str__(self) -> str:
        noise = (
            ""
            if self.noise_kernels is None
            else f", {self.noise_kernels} noise kernels"
        )
        worst = {
            k: f"{v:.3g} (limit {self.limits[k]:.3g})" for k, v in self.worst.items()
        }
        return (
            f"{self.bad} of {self.replays} replays exceed the tolerance{noise}; "
            f"worst error/scale {worst}" + (f"; first {self.first}" if self.bad else "")
        )


class ReplayCheck:
    def __init__(
        self,
        forward: Callable[[], None],
        outputs: Mapping[str, torch.Tensor],
        *,
        reset: Callable[[], None] | None = None,
        device: torch.device | None = None,
        quiet_replays: int = 64,
        margin: float = 8.0,
        floor: float = 0.01,
        broken: float = 0.25,
    ):
        for name, t in outputs.items():
            if not t.is_contiguous():
                raise ValueError(f"output {name} must be contiguous")
        self.device = device or next(iter(outputs.values())).device
        self.outputs = dict(outputs)
        self.names = list(self.outputs)
        self.stream = torch.cuda.Stream(device=self.device)

        def step():
            if reset is not None:
                reset()
            forward()

        with torch.cuda.stream(self.stream):
            step()  # compile and warm outside capture
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            step()
        self._replay()
        self.stream.synchronize()
        self.refs = {n: t.clone() for n, t in self.outputs.items()}
        self.floating = {n: t.is_floating_point() for n, t in self.refs.items()}
        self.scales = {}
        for n, ref in self.refs.items():
            if self.floating[n]:
                finite = ref.float()[torch.isfinite(ref)]
                scale = float(finite.abs().max()) if finite.numel() else 0.0
                if scale == 0.0:
                    raise ValueError(
                        f"output {n}: reference has no nonzero finite values"
                    )
                self.scales[n] = scale

        self.errors = torch.zeros(
            len(self.names), dtype=torch.float32, device=self.device
        )
        self.host = torch.zeros(len(self.names), dtype=torch.float32, pin_memory=True)
        with torch.cuda.stream(self.stream):
            self._compare_ops()
        self.stream.synchronize()
        self.compare = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.compare, stream=self.stream):
            self._compare_ops()
            self.host.copy_(self.errors, non_blocking=True)

        # Calibrate the spread arithmetic order alone produces.
        self.limits = {n: float("inf") for n in self.names}
        quiet = self.run(replays=quiet_replays)
        self.quiet_spread = dict(quiet.worst)
        for n in self.names:
            if self.floating[n]:
                self.limits[n] = max(margin * self.quiet_spread[n], floor)
                if self.quiet_spread[n] > broken:
                    raise NoisyScenario(
                        f"output {n}: quiet replays already differ by "
                        f"{self.quiet_spread[n]:.3g} of the output scale"
                    )
            else:
                self.limits[n] = 0.0
                if self.quiet_spread[n]:
                    raise NoisyScenario(
                        f"output {n}: {int(self.quiet_spread[n])} integer values differ "
                        "between quiet replays; canonicalize them"
                    )

    def _replay(self):
        with torch.cuda.stream(self.stream):
            self.graph.replay()

    def _error(self, n: str) -> torch.Tensor:
        cur, ref = self.outputs[n], self.refs[n]
        if not self.floating[n]:
            return (cur != ref).sum().float()
        cur, ref = cur.float(), ref.float()
        finite = torch.isfinite(ref)
        diff = torch.where(finite, (cur - ref).abs(), torch.zeros_like(ref))
        # NaN compares false; make a broken finiteness pattern infinitely wrong.
        diff = torch.where(
            torch.isfinite(cur) == finite, diff, torch.full_like(ref, float("inf"))
        )
        return torch.nan_to_num(diff, nan=float("inf")).amax() / self.scales[n]

    def _compare_ops(self):
        torch.stack([self._error(n) for n in self.names], out=self.errors)

    def _describe(self, replay: int, row: list[float]) -> dict:
        record = {"replay": replay}
        for n, err in zip(self.names, row):
            if err <= self.limits[n]:
                continue
            cur, ref = self.outputs[n], self.refs[n]
            if self.floating[n]:
                bad = (cur.float() - ref.float()).abs() > self.limits[n] * self.scales[
                    n
                ]
                bad |= torch.isfinite(cur) != torch.isfinite(ref)
            else:
                bad = cur != ref
            entry = {"error_over_scale": round(err, 4), "elements": int(bad.sum())}
            if cur.dim() >= 2:
                rows = bad.reshape(cur.shape[0], -1).any(dim=1).nonzero().view(-1)
                entry["rows"] = rows[:16].tolist()
            record[n] = entry
        return record

    def run(
        self,
        *,
        replays: int,
        seconds: float | None = None,
        contention=None,
        keep_first: int = 4,
    ) -> RaceReport:
        deadline = None if seconds is None else time.monotonic() + seconds
        bad = done = 0
        worst = {n: 0.0 for n in self.names}
        first = []
        start_noise = None if contention is None else contention.launched
        while done < replays and (deadline is None or time.monotonic() < deadline):
            with torch.cuda.stream(self.stream):
                self.graph.replay()
                self.compare.replay()
            self.stream.synchronize()
            done += 1
            row = self.host.tolist()
            for n, err in zip(self.names, row):
                worst[n] = max(worst[n], err)
            if any(err > self.limits[n] for n, err in zip(self.names, row)):
                bad += 1
                if len(first) < keep_first:
                    first.append(self._describe(done, row))
        return RaceReport(
            replays=done,
            bad=bad,
            worst=worst,
            limits=dict(self.limits),
            first=first,
            noise_kernels=None
            if contention is None
            else contention.launched - start_noise,
        )

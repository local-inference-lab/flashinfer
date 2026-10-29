"""Abstract scheduling checks for queued phases, waves, and slot reuse."""

from dataclasses import dataclass, field
import random

import pytest

from b12x.comm.pcie._joint_profile import (
    JOINT_MAX_BLOCKS,
    JOINT_MAX_RANKS,
    JOINT_SIGNAL_BYTES,
    JOINT_QUEUE_SLOT0_OFFSET,
    JOINT_QUEUE_SLOT_STRIDE,
    JOINT_QUEUE_SLOT_BYTES,
    JOINT_QUEUE_CLAIM_OFFSET,
    JOINT_QUEUE_DONE_OFFSET,
    JOINT_QUEUE_MAX_PHASES,
    JOINT_QUEUE_SIGNAL_FIRST_PEER,
    JOINT_QUEUE_ARRIVAL_LOCK,
    joint_modes,
    validate_queue_world,
)


def test_queue_records_are_disjoint_from_every_admitted_peer_signal():
    assert (
        JOINT_QUEUE_CLAIM_OFFSET + JOINT_QUEUE_MAX_PHASES * 8 <= JOINT_QUEUE_DONE_OFFSET
    )
    assert (
        JOINT_QUEUE_DONE_OFFSET + JOINT_QUEUE_MAX_PHASES * 8 <= JOINT_QUEUE_SLOT_BYTES
    )
    for world in (2, 3, 4, 6, 8, 10):
        validate_queue_world(world)
        assert world <= JOINT_QUEUE_SIGNAL_FIRST_PEER
        assert ("queued_push" in joint_modes("joint_latency", world)) == (world > 2)
        for slot in range(2):
            first = JOINT_QUEUE_SLOT0_OFFSET + slot * JOINT_QUEUE_SLOT_STRIDE
            end = first + JOINT_QUEUE_SLOT_BYTES
            assert first % 8 == 0 and end < JOINT_SIGNAL_BYTES - 8
            for block in range(JOINT_MAX_BLOCKS):
                for source in range(world):
                    signal = (
                        JOINT_MAX_BLOCKS * JOINT_MAX_RANKS * 4
                        + slot * JOINT_QUEUE_SLOT_STRIDE
                        + (block * JOINT_MAX_RANKS + source) * 128
                    )
                    assert signal + 4 <= first or signal >= end
    with pytest.raises(ValueError, match="reserved queue"):
        validate_queue_world(12)


@dataclass
class Worker:
    block: int
    generation: int
    state: str = "enter"
    phase: int = 0
    tile: int = 0
    exhausted: set = field(default_factory=set)


@dataclass
class Rank:
    epoch: int = 0
    arrivals: int = 0
    invocation: int = 0
    started: bool = False
    pending: list = field(default_factory=list)
    resident: dict = field(default_factory=dict)
    exited: set = field(default_factory=set)
    controls: list = field(default_factory=lambda: [(0, "unset"), (0, "unset")])
    claims: list = field(default_factory=lambda: [[], []])
    done: list = field(default_factory=lambda: [[], []])
    executed: list = field(default_factory=lambda: [[], []])
    incoming: list = field(default_factory=lambda: [{}, {}])
    reduced: list = field(default_factory=lambda: [{}, {}])
    output: dict = field(default_factory=dict)


def exercise_schedule(world, seed):
    rng = random.Random(seed)
    ranks = [Rank() for _ in range(world)]
    # The gap represents a complete epoch wrap using payload-ready modes, which
    # leave the queue's generation-zero DONE token untouched.
    calls = (64, "gap", 1, 64, "base", 4, 8)
    capacities = [rng.randrange(5) for _ in range(world)]
    capacities[0] = max(1, capacities[0])
    residency = [rng.randrange(1, 5) for _ in range(world)]
    barriers = {}
    base_ready = {}
    coordinators = set()
    phases = 2 * world
    delayed_exhaustions = 0

    def count(rank, phase):
        if phase < world - 1:
            return max(1, capacities[(rank + phase + 1) % world])
        if phase == phases - 1:
            return (world - 1) * max(capacities)
        return max(1, capacities[rank])

    for _ in range(200000):
        if all(rank.invocation == len(calls) for rank in ranks):
            assert any(block != 0 for block in coordinators)
            return delayed_exhaustions
        actions = []
        for r, rank in enumerate(ranks):
            if rank.invocation == len(calls):
                continue
            call = calls[rank.invocation]
            if not rank.started:
                actions.append((r, "start", None))
                continue
            if isinstance(call, str):
                if len(base_ready.get(rank.invocation, ())) == world:
                    actions.append((r, "base", None))
                continue
            if rank.pending and len(rank.resident) < residency[r]:
                actions.append((r, "launch", None))
            for block, worker in rank.resident.items():
                control = rank.controls[worker.generation % 2][1]
                if (
                    worker.state == "enter"
                    and rank.arrivals == JOINT_QUEUE_ARRIVAL_LOCK
                ):
                    continue
                if worker.state == "wait" and control == worker.phase:
                    continue
                if (
                    worker.state == "barrier"
                    and len(barriers.get((rank.invocation, worker.phase), ())) != world
                ):
                    continue
                actions.append((r, worker.state, block))
        assert actions, "all resident workers are waiting with no work owner"
        r, action, block = rng.choice(actions)
        rank = ranks[r]
        invocation = rank.invocation
        call = calls[invocation]
        if action == "start":
            assert rank.arrivals == 0 and not rank.resident
            rank.started = True
            rank.output = {}
            if isinstance(call, str):
                base_ready.setdefault(invocation, set()).add(r)
            else:
                rank.pending = list(range(call))
                rng.shuffle(rank.pending)
                rank.exited = set()
            continue
        if action == "base":
            rank.epoch = (
                rank.epoch + ((1 << 32) - 1 if call == "gap" else 1)
            ) & 0xFFFFFFFF
            rank.started = False
            rank.invocation += 1
            continue
        if action == "launch":
            block = rank.pending.pop()
            rank.resident[block] = Worker(block, rank.epoch)
            continue
        worker = rank.resident[block]
        slot = worker.generation % 2
        phase, tile = worker.phase, worker.tile
        if action == "enter":
            if rank.arrivals == 0:
                rank.arrivals = JOINT_QUEUE_ARRIVAL_LOCK
                worker.state = "initialize"
            else:
                rank.arrivals += 1
                if rank.arrivals == call:
                    rank.arrivals = 0
                    rank.epoch = (rank.epoch + 1) & 0xFFFFFFFF
                worker.state = "load"
        elif action == "initialize":
            rank.controls[slot] = (worker.generation, "initializing")
            rank.claims[slot] = [0] * phases
            rank.done[slot] = [0] * phases
            rank.executed[slot] = [set() for _ in range(phases)]
            worker.state = "initialized"
        elif action == "initialized":
            rank.controls[slot] = (worker.generation, 0)
            rank.arrivals = 1
            if call == 1:
                rank.arrivals = 0
                rank.epoch = (rank.epoch + 1) & 0xFFFFFFFF
            worker.state = "load"
        elif action == "load":
            generation, value = rank.controls[slot]
            assert generation == worker.generation
            worker.phase = value
            worker.state = "exit" if value == phases else "claim"
        elif action == "claim":
            worker.tile = rank.claims[slot][phase]
            rank.claims[slot][phase] += 1
            assert rank.claims[slot][phase] <= count(r, phase) + call
            if worker.tile < count(r, phase):
                worker.state = "work"
            else:
                assert phase not in worker.exhausted
                worker.exhausted.add(phase)
                delayed_exhaustions += rank.controls[slot][1] != phase
                worker.state = "wait"
        elif action == "wait":
            worker.state = "load"
        elif action == "work":
            assert rank.controls[slot] == (worker.generation, phase)
            assert tile not in rank.executed[slot][phase]
            rank.executed[slot][phase].add(tile)
            if phase < world - 1:
                peer = (r + phase + 1) % world
                if tile < capacities[peer]:
                    assert (peer, tile) not in rank.output
                    ranks[peer].incoming[slot][r, tile] = invocation
            elif phase == world - 1:
                if tile < capacities[r]:
                    assert all(
                        rank.incoming[slot][source, tile] == invocation
                        for source in range(world)
                        if source != r
                    )
                    rank.output[r, tile] = invocation
            elif phase < phases - 1:
                peer = (r + phase - world + 1) % world
                if tile < capacities[r]:
                    assert rank.output[r, tile] == invocation
                    ranks[peer].reduced[slot][r, tile] = invocation
            else:
                peer = (r + 1 + tile // max(capacities)) % world
                item = tile % max(capacities)
                if item < capacities[peer]:
                    assert rank.reduced[slot][peer, item] == invocation
                    rank.output[peer, item] = invocation
            worker.state = "complete"
        elif action == "complete":
            rank.done[slot][phase] += 1
            assert rank.done[slot][phase] <= count(r, phase)
            assert (
                rank.done[slot][phase]
                <= rank.claims[slot][phase]
                <= count(r, phase) + call
            )
            worker.state = (
                "coordinate" if rank.done[slot][phase] == count(r, phase) else "load"
            )
        elif action == "coordinate":
            coordinators.add(block)
            if phase in (world - 2, phases - 2):
                barriers.setdefault((invocation, phase), set()).add(r)
                worker.state = "barrier"
            else:
                worker.state = "publish"
        elif action == "barrier":
            worker.state = "publish"
        elif action == "publish":
            assert rank.done[slot][phase] == count(r, phase)
            assert count(r, phase) <= rank.claims[slot][phase] <= count(r, phase) + call
            assert rank.executed[slot][phase] == set(range(count(r, phase)))
            rank.controls[slot] = (worker.generation, phase + 1)
            worker.state = "load"
        elif action == "exit":
            del rank.resident[block]
            rank.exited.add(block)
            if len(rank.exited) == call:
                assert rank.arrivals == 0
                assert all(
                    rank.output[owner, item] == invocation
                    for owner in range(world)
                    for item in range(capacities[owner])
                )
                rank.started = False
                rank.invocation += 1
    raise AssertionError("bounded scheduling model did not finish")


@pytest.mark.parametrize("world", (3, 4, 6, 8, 10))
def test_queued_protocol_progresses_with_delayed_ctas_and_replays(world):
    delayed = 0
    for seed in range(20):
        delayed += exercise_schedule(world, seed)
    assert delayed > 0, "schedules must include stale-phase exhausted claims"

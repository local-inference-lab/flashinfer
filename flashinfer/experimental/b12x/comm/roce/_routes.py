"""Choose which local and remote RDMA device carries each stripe ("rail") to each peer.

On a switched fabric every device reaches every peer, so rail ``h`` simply pairs local device ``h``
with the peer's device ``h``. A switchless topology breaks that: three DGX Sparks cabled in a ring
(port 0 to the next node's port 1) reach each neighbour through one port only, and each port is two
PCIe functions with their own point-to-point subnet. There, rail ``h`` to a peer must use a local
device and a remote device on the same subnet. A fabric can also combine both: DGX Sparks whose
first port joins a switch and whose second port is cabled to one partner reach that partner over
subnets only the two of them hold, and every other peer over the switch's subnets. A subnet's
breadth is the number of ranks holding an address on it: two for a direct cable, the whole world for
a switch.

``plan_routes`` uses index pairing (rail ``h`` on device ``h`` at both ends) when every pair of ranks
shares a subnet on each such pair and every subnet joining two or more ranks joins all of them, which
every purely switched fabric satisfies. Otherwise it pairs devices per peer by subnet. A device without
an IPv4 GID cannot be checked, so both modes trust the caller's device order for it: index pairing
skips the check, and per-peer pairing uses a same-index pair with such a device only when no choice of
verified same-subnet links fills every rail. Per-peer pairing picks the device-disjoint set of links
with the most verified ones and, among those, the narrowest subnets, so a peer with a direct cable is
reached over that cable's functions rather than over a switch. IPv4 link-local subnets
(169.254.0.0/16) count as world-wide: every host using link-local addressing shares that subnet
whether or not its ports are cabled together, so it never identifies a direct link. Every rank runs
the planner over the same exchanged endpoint list, and rails are ordered by subnet breadth and then by
link network address, so both ends of a link always agree on which rail it carries.

A DGX Spark exposes each ConnectX-7 port as two PCIe functions, one behind each of its two PCIe x4
links, and the two functions of one link share its bandwidth. An endpoint records its PCIe link (the
function's PCI address without the function number, for example ``0002:01:00``). Index pairing
requires distinct links among the rail devices of every rank, and per-peer pairing prefers rails on
distinct links at both ends before it compares subnet breadth, so a peer's rails never share a link
when another choice exists. Two ranks cabled to each other and to a switch hold every subnet in
common; their rails then use both functions of one port rather than one function of each port.

Subnet breadth cannot identify the cable between two such ranks, so an endpoint also records its
netdev's MAC address and the MAC addresses its LLDP neighbors advertise as port IDs, read from
``lldpctl`` when lldpd runs on the host. LLDP frames do not cross a bridge, so devices of two ranks
that list each other share a cable. Index pairing requires that no such pair exists, and per-peer
pairing prefers links between LLDP neighbors after PCIe link separation and before subnet breadth;
a two-rank group on a cabled pair then uses one cable function behind each PCIe link. A host
without lldpd records no neighbors and is routed by subnet breadth alone.

``relay_plan`` derives the two-shot all-gather relay (see ``_roce_proxy.c``) for a fabric whose ranks
form pairs joined by direct cables: each rank's partner is the one peer whose rails all use subnets
only the two of them hold. Each pair orders its two ranks by rank number, giving every rank a side.
A rank sends its all-gather chunks directly to its partner and to the rank on its own side in every
other pair; that rank forwards them over its cable to its partner. With two pairs every rank then
sends two of its three all-gather shards over its cable.
"""

from __future__ import annotations

import ipaddress
import itertools
import socket
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

MAX_RAILS = 2
_SIOCGIFNETMASK = 0x891B


@dataclass(frozen=True)
class Endpoint:
    """One local RDMA device: its name, the address with prefix when its GID is IPv4-mapped, the PCIe
    link it shares with sibling functions (PCI domain:bus:device) when known, its netdev's MAC
    address when known, and the MAC addresses its LLDP neighbors advertise as port IDs (sorted,
    lowercase)."""

    name: str
    ipv4: Optional[ipaddress.IPv4Interface]
    link: Optional[str] = None
    mac: Optional[str] = None
    neighbors: tuple[str, ...] = ()


def _pcie_link(device: Path) -> Optional[str]:
    """PCI domain:bus:device of an RDMA device's function, from its sysfs ``device`` link."""
    try:
        address = device.resolve(strict=True).name
    except OSError:
        return None
    base, dot, function = address.rpartition(".")
    return base if dot and base.count(":") == 2 and function.isdigit() else None


def _mac_address(text: str) -> Optional[str]:
    """``text`` as a lowercase MAC address (six colon-separated pairs of hex digits), or None."""
    pairs = text.strip().split(":")
    if len(pairs) != 6 or any(
        len(p) != 2 or any(c not in "0123456789abcdefABCDEF" for c in p) for p in pairs
    ):
        return None
    return ":".join(pairs).lower()


def _parse_lldp(text: str, ifnames: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Per netdev in ``ifnames``, the sorted MAC addresses that ``lldpctl -f keyvalue`` lists as
    neighbor port IDs (``lldp.<netdev>.port.mac=<mac>`` lines)."""
    found: dict[str, set[str]] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if not sep or not key.startswith("lldp."):
            continue
        for ifname in ifnames:
            if key == f"lldp.{ifname}.port.mac":
                mac = _mac_address(value)
                if mac is not None:
                    found.setdefault(ifname, set()).add(mac)
    return {ifname: tuple(sorted(macs)) for ifname, macs in found.items()}


def lldp_neighbors(ifnames: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """LLDP neighbors of ``ifnames`` from lldpd (see ``_parse_lldp``); empty when lldpctl is
    missing, fails, or does not answer within two seconds."""
    if not ifnames:
        return {}
    try:
        done = subprocess.run(
            ["lldpctl", "-f", "keyvalue", *ifnames],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    return _parse_lldp(done.stdout, ifnames) if done.returncode == 0 else {}


def _netmask(ifname: str) -> Optional[str]:
    """IPv4 netmask of ``ifname`` (SIOCGIFNETMASK), or None when unavailable."""
    # Linux-only ioctl; imported here so Windows can import the module.
    import fcntl

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            raw = fcntl.ioctl(
                sock.fileno(),
                _SIOCGIFNETMASK,
                struct.pack("256s", ifname.encode()[:15]),
            )
        except OSError:
            return None
    return socket.inet_ntoa(raw[20:24])


def local_endpoints(
    names: Sequence[str],
    gid_index: int,
    root: Path = Path("/sys/class/infiniband"),
    lldp: Callable[[Sequence[str]], Mapping[str, tuple[str, ...]]] = lldp_neighbors,
) -> tuple[Endpoint, ...]:
    """Describe ``names``: the IPv4 address of each device's GID at ``gid_index`` and its netdev's prefix.

    A device whose GID is not IPv4-mapped, or whose netdev or netmask cannot be read, gets
    ``ipv4=None`` and is treated as reachable by index pairing only. ``link`` is the device's PCIe
    link from its sysfs ``device`` entry, or None when that cannot be read. ``mac`` is the netdev's
    address under the device's PCI function, and ``neighbors`` its LLDP neighbors from ``lldp``.
    """
    ifnames: list[str] = []
    for name in names:
        try:
            ifnames.append(
                (root / name / "ports" / "1" / "gid_attrs" / "ndevs" / str(gid_index))
                .read_text()
                .strip()
            )
        except OSError:
            ifnames.append("")
    neighbors = lldp([i for i in ifnames if i])
    out = []
    for name, netdev in zip(names, ifnames):
        port = root / name / "ports" / "1"
        ipv4 = None
        try:
            raw = bytes.fromhex(
                port.joinpath("gids", str(gid_index))
                .read_text()
                .strip()
                .replace(":", "")
            )
        except (OSError, ValueError):
            raw = b""
        if (
            len(raw) == 16
            and raw[:10] == bytes(10)
            and raw[10:12] == b"\xff\xff"
            and netdev
        ):
            mask = _netmask(netdev)
            if mask is not None:
                ipv4 = ipaddress.IPv4Interface(f"{socket.inet_ntoa(raw[12:])}/{mask}")
        mac = None
        if netdev:
            try:
                mac = _mac_address(
                    (root / name / "device" / "net" / netdev / "address").read_text()
                )
            except OSError:
                mac = None
        out.append(
            Endpoint(
                name,
                ipv4,
                _pcie_link(root / name / "device"),
                mac,
                tuple(neighbors.get(netdev, ())) if netdev else (),
            )
        )
    return tuple(out)


def _same_link(a: Endpoint, b: Endpoint) -> bool:
    """True when both endpoints have IPv4 addresses on the same network."""
    return (
        a.ipv4 is not None and b.ipv4 is not None and a.ipv4.network == b.ipv4.network
    )


def _lldp_adjacent(a: Endpoint, b: Endpoint) -> bool:
    """Whether either device lists the other's MAC address among its LLDP neighbors: LLDP frames do
    not cross a bridge, so the two devices share a cable. Callers compare devices of different ranks."""
    return (b.mac is not None and b.mac in a.neighbors) or (
        a.mac is not None and a.mac in b.neighbors
    )


def _any_lldp_adjacency(endpoints: Sequence[Sequence[Endpoint]]) -> bool:
    """Whether any device of one rank is an LLDP neighbor of a device of another rank."""
    return any(
        _lldp_adjacent(x, y)
        for a in range(len(endpoints))
        for b in range(a + 1, len(endpoints))
        for x in endpoints[a]
        for y in endpoints[b]
    )


def _breadths(
    endpoints: Sequence[Sequence[Endpoint]],
) -> dict[ipaddress.IPv4Network, int]:
    """Per IPv4 network, the number of ranks holding an address on it; a link-local network counts
    as held by every rank, since sharing it does not imply a cable between the holders."""
    world = len(endpoints)
    counts: dict[ipaddress.IPv4Network, int] = {}
    for devices in endpoints:
        for network in {e.ipv4.network for e in devices if e.ipv4 is not None}:
            counts[network] = counts.get(network, 0) + 1
    return {
        network: world if network.is_link_local else count
        for network, count in counts.items()
    }


def _shared_links(devices: Sequence[Endpoint]) -> int:
    """How many of ``devices`` share a known PCIe link with an earlier one."""
    known = [d.link for d in devices if d.link is not None]
    return len(known) - len(set(known))


def _index_pairing_valid(
    endpoints: Sequence[Sequence[Endpoint]],
    rails: int,
    breadths: dict[ipaddress.IPv4Network, int],
) -> bool:
    """True when rail ``h`` can pair device ``h`` with device ``h`` between every pair of ranks, no
    subnet links some ranks but not all of them and no devices of two ranks are LLDP neighbors (a
    direct cable that per-peer pairing would prefer), and no rank's rail devices share a PCIe link."""
    world = len(endpoints)
    if any(2 <= breadth < world for breadth in breadths.values()):
        return False
    if _any_lldp_adjacency(endpoints):
        return False
    if any(len(devices) < rails for devices in endpoints):
        return False
    if any(_shared_links(devices[:rails]) for devices in endpoints):
        return False
    for a in range(len(endpoints)):
        for b in range(a + 1, len(endpoints)):
            for h in range(rails):
                x, y = endpoints[a][h], endpoints[b][h]
                if x.ipv4 is None or y.ipv4 is None:
                    continue  # no IPv4 GID to compare: trust the caller's device order
                if x.ipv4.network != y.ipv4.network:
                    return False
    return True


def _best_matching(
    candidates: list,
    rails: int,
    local: Sequence[Endpoint] = (),
    remote: Sequence[Endpoint] = (),
) -> Optional[list[tuple[int, int]]]:
    """The ``rails`` candidates with distinct local and remote devices that use the most verified links,
    then the fewest rails sharing a PCIe link at either end (``local`` and ``remote`` name the
    devices), then the most links between LLDP neighbors, then the narrowest subnets (smallest total
    breadth), ties broken by the smallest sorted keys; ``None`` when no such set exists.

    ``candidates`` is sorted by key, so each combination, and therefore the rail order, is ordered by key.
    A rank has at most four devices, so the search is at most C(16, 2) combinations.
    """
    best = None
    for combo in itertools.combinations(candidates, rails):
        if (
            len({l for _, l, _ in combo}) < rails
            or len({r for _, _, r in combo}) < rails
        ):
            continue
        shared = (_shared_links([local[l] for _, l, _ in combo]) if local else 0) + (
            _shared_links([remote[r] for _, _, r in combo]) if remote else 0
        )
        score = (
            sum(key[0] for key, _, _ in combo),
            shared,
            sum(key[1] for key, _, _ in combo),
            sum(key[2] for key, _, _ in combo),
            [key for key, _, _ in combo],
        )
        if best is None or score < best[0]:
            best = (score, combo)
    return None if best is None else [(l, r) for _, l, r in best[1]]


def neighbor_ranks(rank: int, world: int, *, ring_open: bool) -> set[int]:
    """The ranks adjacent to ``rank`` in rank order: both around the cycle of a ring, or only the ones
    that exist on a line (``ring_open``), whose end ranks have a single neighbor."""
    if ring_open:
        return {peer for peer in (rank - 1, rank + 1) if 0 <= peer < world}
    return {(rank - 1) % world, (rank + 1) % world}


def plan_routes(
    endpoints: Sequence[Sequence[Endpoint]],
    rank: int,
    rails: int,
    *,
    ring: bool = False,
    ring_open: bool = False,
) -> list[list[tuple[int, int]]]:
    """Per peer, the ``(local_device, remote_device)`` index pair carrying each rail; ``[]`` for ``rank``.

    ``ring`` connects rank-order neighbors on a cycle; ``ring_open`` on a line, where rank 0 and
    rank ``world - 1`` have one neighbor each and the proxy relays through the ranks between."""
    if not 1 <= rails <= MAX_RAILS:
        raise ValueError(f"rails must be 1..{MAX_RAILS}, got {rails}")
    world = len(endpoints)
    if (ring or ring_open) and not 2 <= world <= 16:
        raise ValueError("experimental ring requires 2..16 ranks")
    if not 0 <= rank < world:
        raise ValueError("rank outside the endpoint list")
    if ring or ring_open:
        # Rank order defines the cycle, or the line; the proxy relays all non-neighbors.
        neighbors = neighbor_ranks(rank, world, ring_open=ring_open) | {rank}
        peers = [p for p in range(world) if p in neighbors]
        subset = [endpoints[p] for p in peers]
        planned = plan_routes(subset, peers.index(rank), rails)
        return [planned[peers.index(p)] if p in peers else [] for p in range(world)]
    breadths = _breadths(endpoints)
    if _index_pairing_valid(endpoints, rails, breadths):
        return [
            [] if p == rank else [(h, h) for h in range(rails)] for p in range(world)
        ]
    local = endpoints[rank]
    routes: list[list[tuple[int, int]]] = []
    for peer in range(world):
        if peer == rank:
            routes.append([])
            continue
        remote = endpoints[peer]
        # Candidates: verified same-subnet links, keyed by LLDP adjacency and subnet breadth (a direct
        # cable precedes a switch), network and the link's two addresses; and same-index pairs with no
        # IPv4 GID on either end, which cannot be checked and are trusted in caller order, as index
        # pairing would. Every key is symmetric between the two ranks.
        verified = [
            (
                (
                    0,
                    0 if _lldp_adjacent(local[l], remote[r]) else 1,
                    breadths[local[l].ipv4.network],
                    int(local[l].ipv4.network.network_address),
                    min(int(local[l].ipv4.ip), int(remote[r].ipv4.ip)),
                    max(int(local[l].ipv4.ip), int(remote[r].ipv4.ip)),
                ),
                l,
                r,
            )
            for l in range(len(local))
            for r in range(len(remote))
            if _same_link(local[l], remote[r])
        ]
        unverifiable = [
            ((1, 1, 0, h, 0, 0), h, h)
            for h in range(min(len(local), len(remote)))
            if local[h].ipv4 is None or remote[h].ipv4 is None
        ]
        candidates = sorted(verified + unverifiable)
        chosen = _best_matching(candidates, rails, local, remote)
        if chosen is None:
            reach = next(
                (k for k in range(rails - 1, 0, -1) if _best_matching(candidates, k)), 0
            )
            raise RuntimeError(
                f"rank {rank}: {reach} of {rails} RoCE rails have a link to rank {peer}; "
                f"local {[(e.name, str(e.ipv4)) for e in local]}, "
                f"remote {[(e.name, str(e.ipv4)) for e in remote]}"
            )
        routes.append(chosen)
    return routes


def partners(
    endpoints: Sequence[Sequence[Endpoint]], rails: int
) -> list[Optional[int]]:
    """Per rank, its only peer whose every rail uses a subnet the two of them alone hold, else
    ``None``. With two ranks every subnet joins both, so no rank has a partner."""
    world = len(endpoints)
    if world < 3:
        return [None] * world
    breadths = _breadths(endpoints)
    out: list[Optional[int]] = []
    for rank in range(world):
        local = endpoints[rank]
        direct = [
            peer
            for peer, links in enumerate(plan_routes(endpoints, rank, rails))
            if links
            and all(
                local[l].ipv4 is not None and breadths[local[l].ipv4.network] == 2
                for l, _ in links
            )
        ]
        out.append(direct[0] if len(direct) == 1 else None)
    return out


def relay_plan(
    endpoints: Sequence[Sequence[Endpoint]], rank: int, rails: int
) -> Optional[list[int]]:
    """The all-gather relay plan of ``rank`` for ``Proxy.set_relay``, or ``None`` unless the ranks
    form at least two pairs of mutual partners and no rank is left out.

    Entry ``p`` names the rank that forwards ``rank``'s chunks to rank ``p`` (-1: sent directly) and
    entry ``world + s`` the rank to which ``rank`` forwards rank ``s``'s chunks (-1: none).
    """
    world = len(endpoints)
    mates = partners(endpoints, rails)
    if (
        world < 4
        or world % 2
        or any(m is None or mates[m] != r for r, m in enumerate(mates))
    ):
        return None
    side = [0 if r < mates[r] else 1 for r in range(world)]
    mate = mates[rank]
    via = [-1] * world
    forward = [-1] * world
    for peer in range(world):
        if peer in (rank, mate):
            continue
        if side[peer] != side[rank]:
            # Peer's partner is on our side and receives our chunks directly.
            via[peer] = mates[peer]
        else:
            # Peer is on our side: we receive its chunks and pass them to our partner.
            forward[peer] = mate
    return via + forward


__all__ = [
    "Endpoint",
    "MAX_RAILS",
    "local_endpoints",
    "partners",
    "plan_routes",
    "relay_plan",
]

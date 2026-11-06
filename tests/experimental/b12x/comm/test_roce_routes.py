"""Per-peer RoCE route planning: switched fabrics keep index pairing, switchless rings pair by subnet."""

import dataclasses
import ipaddress

import pytest

from b12x.comm.roce._routes import Endpoint, partners, plan_routes, relay_plan


def ep(name, iface=None, link=None):
    """Endpoint named ``name`` with an optional IPv4 interface such as ``10.0.0.1/30`` and PCIe link."""
    return Endpoint(
        name, None if iface is None else ipaddress.IPv4Interface(iface), link
    )


# DGX Spark ConnectX-7 functions: each PCIe x4 link carries one function of each port.
SPARK_LINKS = {
    "roceP2p1s0f0": "0002:01:00",
    "roceP2p1s0f1": "0002:01:00",
    "rocep1s0f0": "0000:01:00",
    "rocep1s0f1": "0000:01:00",
}


def spark(name, iface):
    return ep(name, iface, SPARK_LINKS[name])


# Three DGX Sparks cabled port 0 -> next node's port 1 (no switch). Every cable shows up as two
# PCIe functions, each with its own /30, so each peer is reachable over exactly two local devices.
RING = [
    [
        ep("rocep1s0f0", "10.10.0.1/30"),
        ep("rocep1s0f1", "10.10.4.2/30"),
        ep("roceP2p1s0f0", "10.10.1.1/30"),
        ep("roceP2p1s0f1", "10.10.5.2/30"),
    ],
    [
        ep("rocep1s0f0", "10.10.2.1/30"),
        ep("rocep1s0f1", "10.10.0.2/30"),
        ep("roceP2p1s0f0", "10.10.3.1/30"),
        ep("roceP2p1s0f1", "10.10.1.2/30"),
    ],
    [
        ep("rocep1s0f0", "10.10.4.1/30"),
        ep("rocep1s0f1", "10.10.2.2/30"),
        ep("roceP2p1s0f0", "10.10.5.1/30"),
        ep("roceP2p1s0f1", "10.10.3.2/30"),
    ],
]


def _link(endpoints, rank, peer, local, remote):
    """True when the two chosen devices share an IPv4 network."""
    a = endpoints[rank][local].ipv4
    b = endpoints[peer][remote].ipv4
    return a.network == b.network


def test_ring_routes_every_rail_over_a_real_link():
    """Every rail to every peer uses two devices on the same /30, and a peer's rails use distinct devices."""
    rails = 2
    for rank in range(3):
        routes = plan_routes(RING, rank, rails)
        assert routes[rank] == []
        for peer in range(3):
            if peer == rank:
                continue
            assert len(routes[peer]) == rails
            assert len({l for l, _ in routes[peer]}) == rails
            for local, remote in routes[peer]:
                assert _link(RING, rank, peer, local, remote)


def test_ring_routes_are_symmetric_so_both_ends_use_the_same_link_per_rail():
    """Rank a's route to b is rank b's route to a with the ends swapped, rail for rail."""
    plans = [plan_routes(RING, rank, 2) for rank in range(3)]
    for a in range(3):
        for b in range(3):
            if a != b:
                assert plans[a][b] == [(r, l) for l, r in plans[b][a]]


def test_ring_rank0_uses_port0_functions_to_the_next_node_and_port1_functions_to_the_previous():
    """On the ring, port 0's two functions reach the next node and port 1's reach the previous one."""
    routes = plan_routes(RING, 0, 2)
    names = RING[0]
    assert {names[l].name for l, _ in routes[1]} == {"rocep1s0f0", "roceP2p1s0f0"}
    assert {names[l].name for l, _ in routes[2]} == {"rocep1s0f1", "roceP2p1s0f1"}


def test_switched_fabric_keeps_index_pairing():
    """Two subnets shared by every rank: rail h uses device h at both ends."""
    switched = [
        [ep("rocep1s0f0", "192.168.42.10/24"), ep("roceP2p1s0f0", "192.168.43.10/24")],
        [ep("rocep1s0f0", "192.168.42.11/24"), ep("roceP2p1s0f0", "192.168.43.11/24")],
        [ep("rocep1s0f0", "192.168.42.12/24"), ep("roceP2p1s0f0", "192.168.43.12/24")],
    ]
    for rank in range(3):
        routes = plan_routes(switched, rank, 2)
        for peer in range(3):
            if peer != rank:
                assert routes[peer] == [(0, 0), (1, 1)]


def test_one_flat_subnet_with_extra_devices_keeps_index_pairing_on_the_first_rails():
    """Four devices on one subnet: the rails use devices 0 and 1 at both ends."""
    flat = [
        [ep(f"d{h}", f"10.0.0.{10 * r + h}/24") for h in range(4)] for r in range(2)
    ]
    assert plan_routes(flat, 0, 2)[1] == [(0, 0), (1, 1)]


def test_unknown_addresses_fall_back_to_index_pairing():
    """Devices without IPv4 GIDs are paired in the order given."""
    unknown = [[ep("mlx5_0"), ep("mlx5_1")], [ep("mlx5_0"), ep("mlx5_1")]]
    assert plan_routes(unknown, 1, 2)[0] == [(0, 0), (1, 1)]


def test_two_node_direct_cable_pairs_the_matching_functions():
    """A two-node direct cable pairs each function with the one on its subnet, whatever the device order."""
    direct = [
        [ep("rocep1s0f0", "10.20.0.1/30"), ep("roceP2p1s0f0", "10.20.1.1/30")],
        [ep("roceP2p1s0f0", "10.20.1.2/30"), ep("rocep1s0f0", "10.20.0.2/30")],
    ]
    assert plan_routes(direct, 0, 2)[1] == [(0, 1), (1, 0)]
    assert plan_routes(direct, 1, 2)[0] == [(1, 0), (0, 1)]


def test_single_rail_uses_one_link_per_peer():
    """With one rail, each peer gets one route over a real link."""
    routes = plan_routes(RING, 1, 1)
    for peer in (0, 2):
        ((local, remote),) = routes[peer]
        assert _link(RING, 1, peer, local, remote)


def test_peer_with_too_few_links_is_a_clear_error():
    """A peer reachable over fewer links than rails fails with the ranks and counts named."""
    broken = [list(r) for r in RING]
    # Rank 2 loses roceP2p1s0f1 (10.10.3.2), one of its two links to rank 1.
    broken[2] = [e for e in RING[2] if e.name != "roceP2p1s0f1"]
    with pytest.raises(RuntimeError, match="rank 1: 1 of 2 .* rank 2"):
        plan_routes(broken, 1, 2)


def test_fallback_keeps_an_unverifiable_same_index_pair_after_verified_links():
    """Subnet fallback still trusts a same-index pair it cannot check, but only after verified links."""
    mixed = [
        [ep("ib0"), ep("roce1", "10.0.0.1/30"), ep("roce2", "10.0.9.1/30")],
        [ep("ib0"), ep("roce1", "10.0.5.2/30"), ep("roce2", "10.0.0.2/30")],
    ]
    # Device 1 pairs with device 1 on different subnets, so index pairing is invalid; the only
    # verified link is rank 0 device 1 to rank 1 device 2, and device 0 on both ends has no IPv4 GID.
    assert plan_routes(mixed, 0, 2)[1] == [(1, 2), (0, 0)]
    assert plan_routes(mixed, 1, 2)[0] == [(2, 1), (0, 0)]


def test_fallback_prefers_verified_links_over_unverifiable_pairs():
    """With enough verified links, an unverifiable same-index pair is not used."""
    ring = [list(r) for r in RING]
    ring[0] = [ep("mlx5_0")] + ring[0]
    ring[1] = [ep("mlx5_0")] + ring[1]
    ring[2] = [ep("mlx5_0")] + ring[2]
    for rank in range(3):
        for peer, links in enumerate(plan_routes(ring, rank, 2)):
            assert (0, 0) not in links


def test_fallback_finds_a_complete_matching_a_greedy_pick_would_miss():
    """A verified link that blocks the only unverifiable pair is skipped for one that leaves room for it."""
    local = [ep("roce0", "10.0.0.1/24"), ep("ib1")]
    remote = [
        ep("roce0", "10.0.9.2/24"),
        ep("roce1", "10.0.0.2/24"),
        ep("roce2", "10.0.0.3/24"),
    ]
    # Taking (0, 1) first would leave (1, 1) unusable; (0, 2) plus the trusted (1, 1) supplies both rails.
    assert plan_routes([local, remote], 0, 2)[1] == [(0, 2), (1, 1)]
    assert plan_routes([local, remote], 1, 2)[0] == [(2, 0), (1, 1)]


def test_fallback_tie_break_is_symmetric_on_a_shared_subnet():
    """Two equally verified matchings on one flat subnet: both ends still choose the same links."""
    a = [ep("d0", "10.1.0.1/24"), ep("d1", "10.1.0.2/24"), ep("x", "10.9.0.1/24")]
    b = [ep("d0", "10.1.0.5/24"), ep("d1", "10.1.0.6/24"), ep("y", "10.8.0.1/24")]
    # Index pairing fails on device 2, so both ranks fall back to subnet matching.
    forward = plan_routes([a, b], 0, 2)[1]
    backward = plan_routes([a, b], 1, 2)[0]
    assert backward == [(r, l) for l, r in forward]


# Four DGX Sparks whose first port joins a switch (two functions on 10.200.0.0/24 and 10.200.1.0/24)
# and whose second port is cabled to one partner: ranks 1 and 2 over 10.200.2.0/24 and 10.200.3.0/24,
# ranks 0 and 3 over 10.200.4.0/24 and 10.200.5.0/24. Devices are listed in discovery (name) order.
PARTNER = {0: 3, 1: 2, 2: 1, 3: 0}
SWITCH_DEVICES = {"roceP2p1s0f0", "rocep1s0f0"}
DIRECT_DEVICES = {"roceP2p1s0f1", "rocep1s0f1"}


def _switch_with_partner_cables():
    endpoints = []
    for rank in range(4):
        host = 11 + rank
        direct = 2 if rank in (1, 2) else 4
        endpoints.append(
            [
                spark("roceP2p1s0f0", f"10.200.0.{host}/24"),
                spark("roceP2p1s0f1", f"10.200.{direct}.{host}/24"),
                spark("rocep1s0f0", f"10.200.1.{host}/24"),
                spark("rocep1s0f1", f"10.200.{direct + 1}.{host}/24"),
            ]
        )
    return endpoints


def _route_devices(endpoints, rank, links):
    return {endpoints[rank][local].name for local, _ in links}


@pytest.mark.parametrize("switch_first", [False, True])
def test_partner_cable_carries_both_rails_to_the_partner_and_the_switch_the_rest(
    switch_first,
):
    """Each rank reaches its partner over both cable functions and every other peer over the switch.

    Listing the switch functions first would make index pairing valid; the cable still wins.
    """
    endpoints = _switch_with_partner_cables()
    if switch_first:
        endpoints = [
            sorted(devices, key=lambda e: e.name not in SWITCH_DEVICES)
            for devices in endpoints
        ]
    plans = [plan_routes(endpoints, rank, 2) for rank in range(4)]
    for rank in range(4):
        for peer in range(4):
            if peer == rank:
                assert plans[rank][peer] == []
                continue
            links = plans[rank][peer]
            expected = DIRECT_DEVICES if peer == PARTNER[rank] else SWITCH_DEVICES
            assert _route_devices(endpoints, rank, links) == expected
            assert all(_link(endpoints, rank, peer, a, b) for a, b in links)
            assert links == [(b, a) for a, b in plans[peer][rank]]


def test_partner_cable_down_at_one_end_moves_that_pair_to_the_switch():
    """A partner whose cable functions are absent is reached over the switch from both ends."""
    endpoints = _switch_with_partner_cables()
    endpoints[2] = [e for e in endpoints[2] if e.name in SWITCH_DEVICES]
    plans = [plan_routes(endpoints, rank, 2) for rank in range(4)]
    assert _route_devices(endpoints, 1, plans[1][2]) == SWITCH_DEVICES
    assert plans[1][2] == [(b, a) for a, b in plans[2][1]]
    assert _route_devices(endpoints, 0, plans[0][3]) == DIRECT_DEVICES
    assert _route_devices(endpoints, 3, plans[3][0]) == DIRECT_DEVICES


def test_link_local_addresses_on_two_uncabled_ports_do_not_attract_rails():
    """Two ranks with IPv4 link-local addresses share 169.254.0.0/16 without a cable between them."""
    endpoints = [
        [ep("roceP2p1s0f0", "10.200.0.14/24"), ep("rocep1s0f0", "10.200.1.14/24")],
        [
            ep("roceP2p1s0f0", "10.200.0.12/24"),
            ep("rocep1s0f0", "10.200.1.12/24"),
            ep("rocep1s0f1", "169.254.210.217/16"),
        ],
        [ep("roceP2p1s0f0", "10.200.0.13/24"), ep("rocep1s0f0", "10.200.1.13/24")],
        [
            ep("roceP2p1s0f0", "10.200.0.11/24"),
            ep("rocep1s0f0", "10.200.1.11/24"),
            ep("rocep1s0f1", "169.254.21.112/16"),
        ],
    ]
    for rank in range(4):
        for peer, links in enumerate(plan_routes(endpoints, rank, 2)):
            assert links == ([] if peer == rank else [(0, 0), (1, 1)])


def test_a_cabled_pair_keeps_its_two_rails_on_separate_pcie_links():
    """Two ranks cabled to each other and to a switch hold every subnet in common, and without LLDP
    neighbors nothing identifies the cable. Index pairing in name order would put both rails behind
    one PCIe x4 link; the planner uses one function of each."""
    pair = [_switch_with_partner_cables()[r] for r in (0, 3)]
    for rank in range(2):
        links = plan_routes(pair, rank, 2)[1 - rank]
        assert {SPARK_LINKS[pair[rank][l].name] for l, _ in links} == {
            "0002:01:00",
            "0000:01:00",
        }
        assert _route_devices(pair, rank, links) == SWITCH_DEVICES
    assert plan_routes(pair, 0, 2)[1] == [(b, a) for a, b in plan_routes(pair, 1, 2)[0]]
    # Without PCIe link information the planner keeps device order.
    unknown = [[ep(e.name, str(e.ipv4)) for e in devices] for devices in pair]
    assert plan_routes(unknown, 0, 2)[1] == [(0, 0), (1, 1)]


def _with_lldp(endpoints, partner):
    """``endpoints`` with the LLDP view of DGX Sparks whose cable joins rank ``r`` to ``partner[r]``:
    each function has its own MAC address and lists its sibling function on the same port; a switch
    function also lists the switch, and a cable function both functions of the partner's cable port."""

    def mac(rank, device):
        return f"02:00:00:00:{rank:02x}:{device:02x}"

    def port(rank, cable):
        return [
            mac(rank, d)
            for d, e in enumerate(endpoints[rank])
            if (e.name in DIRECT_DEVICES) == cable
        ]

    out = []
    for rank, devices in enumerate(endpoints):
        row = []
        for d, e in enumerate(devices):
            cable = e.name in DIRECT_DEVICES
            neighbors = [m for m in port(rank, cable) if m != mac(rank, d)]
            neighbors += port(partner[rank], True) if cable else ["02:00:00:00:ff:00"]
            row.append(
                dataclasses.replace(
                    e, mac=mac(rank, d), neighbors=tuple(sorted(neighbors))
                )
            )
        out.append(row)
    return out


def test_lldp_neighbors_route_a_two_rank_cabled_pair_over_its_cable():
    """A two-rank group on a cabled pair holds every subnet in common; LLDP identifies the cable,
    which then carries both rails, one function behind each PCIe link."""
    full = _with_lldp(_switch_with_partner_cables(), PARTNER)
    pair = [full[0], full[3]]
    plans = [plan_routes(pair, rank, 2) for rank in range(2)]
    for rank in range(2):
        links = plans[rank][1 - rank]
        assert _route_devices(pair, rank, links) == DIRECT_DEVICES
        assert {SPARK_LINKS[pair[rank][l].name] for l, _ in links} == {
            "0002:01:00",
            "0000:01:00",
        }
        assert links == [(b, a) for a, b in plans[1 - rank][rank]]
    assert partners(pair, 2) == [None, None]
    assert relay_plan(pair, 0, 2) is None


@pytest.mark.parametrize("fabric", ["partner_cables", "paired_switch"])
def test_lldp_neighbors_leave_routes_planned_by_breadth_unchanged(fabric):
    if fabric == "partner_cables":
        endpoints, partner = _switch_with_partner_cables(), PARTNER
    else:
        endpoints, partner = _paired_switch(2), {0: 1, 1: 0, 2: 3, 3: 2}
    lldp = _with_lldp(endpoints, partner)
    for rank in range(len(endpoints)):
        assert plan_routes(lldp, rank, 2) == plan_routes(endpoints, rank, 2)
        assert relay_plan(lldp, rank, 2) == relay_plan(endpoints, rank, 2)
    assert partners(lldp, 2) == partners(endpoints, 2)


def _paired_switch(pairs):
    """``2 * pairs`` ranks on one switch; ranks 2i and 2i + 1 are also cabled to each other."""
    endpoints = []
    for rank in range(2 * pairs):
        host, pair = 11 + rank, rank // 2
        endpoints.append(
            [
                spark("roceP2p1s0f0", f"10.200.0.{host}/24"),
                spark("roceP2p1s0f1", f"10.201.{2 * pair}.{host}/24"),
                spark("rocep1s0f0", f"10.200.1.{host}/24"),
                spark("rocep1s0f1", f"10.201.{2 * pair + 1}.{host}/24"),
            ]
        )
    return endpoints


def _relay_writers(plans):
    """Per (source, receiver), the ranks that write the source's all-gather chunks to the receiver."""
    world = len(plans)
    writers = {}
    for source in range(world):
        for peer in range(world):
            if peer != source and plans[source][peer] == -1:
                writers.setdefault((source, peer), []).append(source)
    for forwarder in range(world):
        for source in range(world):
            target = plans[forwarder][world + source]
            if target != -1:
                # A forwarder receives the source's chunks directly.
                assert writers.get((source, forwarder)) == [source]
                writers.setdefault((source, target), []).append(forwarder)
    return writers


@pytest.mark.parametrize(
    "endpoints, mates",
    [
        (_switch_with_partner_cables(), PARTNER),
        (_paired_switch(4), {r: r ^ 1 for r in range(8)}),
    ],
    ids=["four-ranks", "eight-ranks"],
)
def test_relay_delivers_every_shard_once_and_forwards_only_over_cables(
    endpoints, mates
):
    """Every rank receives every other rank's all-gather chunks from exactly one writer: the
    source itself, or the rank that its plan names for that receiver, which is the receiver's
    partner. Each rank forwards only to its partner."""
    world = len(endpoints)
    assert partners(endpoints, 2) == [mates[r] for r in range(world)]
    plans = [relay_plan(endpoints, rank, 2) for rank in range(world)]
    writers = _relay_writers(plans)
    for source in range(world):
        for peer in range(world):
            if peer == source:
                continue
            (writer,) = writers[(source, peer)]
            assert writer in (source, plans[source][peer])
            if writer != source:
                assert writer == mates[peer]
    for rank in range(world):
        assert {t for t in plans[rank][world:] if t != -1} <= {mates[rank]}


def test_relay_puts_two_of_three_all_gather_shards_on_the_cable_with_two_pairs():
    """With two pairs, each rank sends its shard to its partner and to one rank of the other pair,
    and forwards one shard to its partner: two cable sends and one switch send."""
    endpoints = _switch_with_partner_cables()
    for rank in range(4):
        plan = relay_plan(endpoints, rank, 2)
        direct = [p for p in range(4) if p != rank and plan[p] == -1]
        forwarded = [t for t in plan[4:] if t != -1]
        sends = direct + forwarded
        assert sends.count(PARTNER[rank]) == 2 and len(sends) == 3


def test_relay_plan_needs_every_rank_in_a_cable_pair():
    """No plan on a switched fabric, with a pair's cable down at one end, or with two ranks."""
    switched = [
        [
            spark("roceP2p1s0f0", f"10.200.0.{11 + r}/24"),
            spark("rocep1s0f0", f"10.200.1.{11 + r}/24"),
        ]
        for r in range(4)
    ]
    assert partners(switched, 2) == [None] * 4
    assert relay_plan(switched, 0, 2) is None
    endpoints = _switch_with_partner_cables()
    endpoints[2] = [e for e in endpoints[2] if e.name in SWITCH_DEVICES]
    assert partners(endpoints, 2) == [3, None, None, 0]
    assert all(relay_plan(endpoints, rank, 2) is None for rank in range(4))
    pair = [_switch_with_partner_cables()[r] for r in (0, 3)]
    assert partners(pair, 2) == [None, None]
    assert relay_plan(pair, 0, 2) is None


def test_a_switched_fabric_with_both_functions_of_a_link_listed_first_avoids_the_shared_link():
    """Index pairing is rejected when a rank's first two devices share a PCIe link."""
    switched = [
        [
            spark("roceP2p1s0f0", f"10.1.0.{r}/24"),
            spark("roceP2p1s0f1", f"10.1.1.{r}/24"),
            spark("rocep1s0f0", f"10.1.2.{r}/24"),
            spark("rocep1s0f1", f"10.1.3.{r}/24"),
        ]
        for r in range(1, 4)
    ]
    for rank in range(3):
        for peer, links in enumerate(plan_routes(switched, rank, 2)):
            if peer != rank:
                assert len({SPARK_LINKS[switched[rank][l].name] for l, _ in links}) == 2


def test_rails_out_of_range_rejected():
    """Rail counts outside 1..MAX_RAILS are rejected."""
    with pytest.raises(ValueError):
        plan_routes(RING, 0, 0)
    with pytest.raises(ValueError):
        plan_routes(RING, 0, 3)


@pytest.mark.parametrize(
    "gid,netmask,expected",
    [
        ("0000:0000:0000:0000:0000:ffff:0a0a:0001", "255.255.255.252", "10.10.0.1/30"),
        ("0000:0000:0000:0000:0000:ffff:0a0a:0001", None, None),
        ("fe80:0000:0000:0000:0000:0000:0000:0001", "255.255.255.252", None),
        ("invalid", "255.255.255.252", None),
    ],
)
def test_local_endpoints_uses_gid_netdev_prefix(
    tmp_path, monkeypatch, gid, netmask, expected
):
    from b12x.comm.roce import _routes

    port = tmp_path / "rdma0" / "ports" / "1"
    (port / "gids").mkdir(parents=True)
    (port / "gid_attrs" / "ndevs").mkdir(parents=True)
    (port / "gids" / "3").write_text(gid)
    (port / "gid_attrs" / "ndevs" / "3").write_text("eth7")
    seen = []

    def mask(name):
        seen.append(name)
        return netmask

    monkeypatch.setattr(_routes, "_netmask", mask)
    function = tmp_path / "pci" / "0002:01:00.1"
    (function / "net" / "eth7").mkdir(parents=True)
    (function / "net" / "eth7" / "address").write_text("4C:BB:47:2C:1E:AB\n")
    (tmp_path / "rdma0" / "device").symlink_to(function)
    asked = []

    def lldp(ifnames):
        asked.extend(ifnames)
        return {"eth7": ("4c:bb:47:7f:c3:c3",)}

    actual = _routes.local_endpoints(("rdma0", "missing"), 3, root=tmp_path, lldp=lldp)
    rdma0 = dataclasses.replace(
        ep("rdma0", expected, "0002:01:00"),
        mac="4c:bb:47:2c:1e:ab",
        neighbors=("4c:bb:47:7f:c3:c3",),
    )
    assert actual == (rdma0, ep("missing"))
    assert asked == ["eth7"]
    assert seen == (["eth7"] if "ffff" in gid else [])


def test_lldp_neighbors_are_the_port_mac_addresses_listed_per_netdev():
    from b12x.comm.roce._routes import _mac_address, _parse_lldp

    text = "\n".join(
        [
            "lldp.enp1s0f1np1.via=LLDP",
            "lldp.enp1s0f1np1.chassis.mac=4c:bb:47:7f:c3:c1",
            "lldp.enp1s0f1np1.port.mac=4C:BB:47:7F:C3:C3",
            "lldp.enp1s0f1np1.port.mac=4c:bb:47:2c:1e:ab",
            "lldp.enp1s0f1np1.port.mac=4c:bb:47:7f:c3:c3",
            "lldp.enp1s0f0np0.port.ifname=qsfp56-dd-1-1",
            "lldp.enp1s0f1np1.100.port.mac=02:00:00:00:00:01",
            "lldp.other.port.mac=02:00:00:00:00:02",
        ]
    )
    assert _parse_lldp(text, ["enp1s0f1np1", "enp1s0f0np0"]) == {
        "enp1s0f1np1": ("4c:bb:47:2c:1e:ab", "4c:bb:47:7f:c3:c3")
    }
    assert _mac_address("4c:bb:47:2c:1e") is None
    assert _mac_address("4c:bb:47:2c:1e:zz") is None


@pytest.mark.parametrize("switched", [False, True])
def test_four_rank_ring_opens_only_neighbors_and_mirrors_every_rail(switched):
    def rank(r):
        if switched:
            return [ep("f0", f"10.1.0.{r + 1}/24"), ep("f1", f"10.2.0.{r + 1}/24")]
        previous = (r + 3) % 4
        return [
            ep("p0f0", f"10.{r + 1}.0.1/30"),
            ep("p0f1", f"10.{r + 1}.1.1/30"),
            ep("p1f0", f"10.{previous + 1}.0.2/30"),
            ep("p1f1", f"10.{previous + 1}.1.2/30"),
        ]

    endpoints = [rank(r) for r in range(4)]
    # Device enumeration need not be identical across hosts.
    endpoints[1].reverse()
    endpoints[3] = endpoints[3][1:] + endpoints[3][:1]
    plans = [plan_routes(endpoints, r, 2, ring=True) for r in range(4)]
    for r in range(4):
        assert plans[r][r] == plans[r][(r + 2) % 4] == []
        for peer in ((r + 1) % 4, (r + 3) % 4):
            assert len(plans[r][peer]) == 2
            assert plans[r][peer] == [(b, a) for a, b in plans[peer][r]]
            assert all(_link(endpoints, r, peer, a, b) for a, b in plans[r][peer])
    if not switched:
        with pytest.raises(RuntimeError, match="0 of 2"):
            plan_routes(endpoints, 0, 2)


@pytest.mark.parametrize("world", [0, 1, 17])
def test_experimental_ring_refuses_other_world_sizes(world):
    with pytest.raises(ValueError, match="2..16"):
        plan_routes([[ep("f0", "10.0.0.1/24")]] * world, 0, 1, ring=True)


@pytest.mark.parametrize("world", [2, 3, 5, 6, 8, 16])
def test_ring_planning_uses_the_configured_rank_count(world):
    endpoints = [
        [ep("f0", f"10.1.0.{r + 1}/24"), ep("f1", f"10.2.0.{r + 1}/24")]
        for r in range(world)
    ]
    for rank in range(world):
        routes = plan_routes(endpoints, rank, 2, ring=True)
        assert len(routes) == world
        assert {p for p, links in enumerate(routes) if links} == {
            (rank - 1) % world,
            (rank + 1) % world,
        }

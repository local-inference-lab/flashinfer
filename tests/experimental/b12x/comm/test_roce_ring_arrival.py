"""A later packet cannot be published while its lane predecessor is missing.

The driver drives one lane window of the production relay on both rails and in
each of the three receive roles: last unicast hop, chain endpoint, and an
intermediate fanout hop that forwards downstream.
"""

import re
import subprocess
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "flashinfer/experimental/b12x/comm/roce/_roce_proxy.c"
)

# The driver wraps one lane window of an open chain, so its serials are derived from the
# proxy's own RING_OPEN_WINDOW rather than pinned to the default. The first window, the
# two window boundaries and a third wrap cover slot straddling and repeated reuse of one slot.
WINDOW_DEFINE = re.search(
    r"^#define RING_OPEN_WINDOW (\d+)$", SOURCE.read_text(), re.MULTILINE
)
assert WINDOW_DEFINE is not None, "the proxy must define RING_OPEN_WINDOW"
RING_WINDOW = int(WINDOW_DEFINE.group(1))
SERIALS = sorted(
    {RING_WINDOW, 3 * RING_WINDOW, 3 * RING_WINDOW + 1, 3 * RING_WINDOW + 2}
    | set(range(RING_WINDOW + 1, RING_WINDOW + 9))
)


@pytest.mark.parametrize("first_serial", SERIALS)
@pytest.mark.parametrize(
    "scan", [1, 2, 3], ids=["during-scan", "after-a-pass", "ack-reuse"]
)
@pytest.mark.parametrize(
    "mode", [0, 1, 2], ids=["final-hop", "endpoint", "intermediate"]
)
@pytest.mark.parametrize("rail", [0, 1], ids=["rail0", "rail1"])
def test_later_packet_cannot_publish_before_its_predecessor(
    proxy_driver, first_serial, scan, mode, rail
):
    if scan == 3 and first_serial == RING_WINDOW:
        pytest.skip("the ack-reuse trigger needs a retired predecessor serial")
    result = subprocess.run(
        [str(proxy_driver), *map(str, (first_serial, scan, mode, rail))],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ordered relay arrival: ok"

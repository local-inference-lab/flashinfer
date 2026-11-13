"""What a relayed packet needs from the queue pair, and where it may go.

The driver links the production proxy against a fake verbs provider that enforces
the declared scatter-gather cap and accounts every write by where it lands: the
two-SGE packet and the queue-pair cap, the startup mailbox size, the terminal
context, and the fanout guard that starts at four ranks.
"""

import subprocess


def test_relay_guard(proxy_driver):
    result = subprocess.run(
        [str(proxy_driver)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "relay guard: ok"

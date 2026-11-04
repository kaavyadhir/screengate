"""Tests for the public-demo rate limiter.

The interesting case is not "does it count to 30". It is whether a
client can escape the limit by controlling its own headers, which is the
bug the obvious X-Forwarded-For implementation has.
"""

from __future__ import annotations

from screengate.ratelimit import SlidingWindowLimiter, client_key


class _FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class _FakeRequest:
    """Minimal stand-in: client_key only reads .headers and .client."""

    def __init__(self, headers: dict[str, str], peer: str | None = "10.0.0.1") -> None:
        self.headers = {k.lower(): v for k, v in headers.items()}
        self.client = _FakeClient(peer) if peer else None


def test_allows_up_to_the_limit_then_blocks():
    lim = SlidingWindowLimiter(events=3, window_s=60.0)
    assert [lim.check("a", now=0.0)[0] for _ in range(3)] == [True, True, True]
    allowed, retry_after = lim.check("a", now=0.0)
    assert allowed is False
    assert 0.0 < retry_after <= 60.0


def test_window_slides():
    lim = SlidingWindowLimiter(events=2, window_s=10.0)
    lim.check("a", now=0.0)
    lim.check("a", now=1.0)
    assert lim.check("a", now=2.0)[0] is False
    # The first event ages out at t=10, freeing one slot.
    assert lim.check("a", now=10.5)[0] is True


def test_keys_are_independent():
    lim = SlidingWindowLimiter(events=1, window_s=60.0)
    assert lim.check("a", now=0.0)[0] is True
    assert lim.check("a", now=0.0)[0] is False
    assert lim.check("b", now=0.0)[0] is True


def test_key_table_is_bounded():
    """An attacker rotating source addresses must not grow memory forever."""
    lim = SlidingWindowLimiter(events=1, window_s=1.0, max_keys=50)
    for i in range(500):
        lim.check(f"host-{i}", now=float(i))
    assert len(lim._hits) <= 50


def test_client_key_uses_socket_address_when_no_proxy():
    assert client_key(_FakeRequest({}, peer="203.0.113.9")) == "203.0.113.9"


def test_client_key_trusts_the_rightmost_forwarded_entry():
    """This is the security property, not a formatting detail.

    A client can put anything in X-Forwarded-For. Caddy appends the
    address it actually observed, so only the LAST entry is trustworthy.
    Reading the first entry - the usual convention - would let a caller
    pick a new identity per request and never be limited.
    """
    spoofed = _FakeRequest(
        {"X-Forwarded-For": "1.2.3.4, 5.6.7.8, 198.51.100.7"}, peer="172.18.0.2"
    )
    assert client_key(spoofed) == "198.51.100.7"


def test_spoofed_forwarded_header_cannot_escape_the_limit():
    lim = SlidingWindowLimiter(events=2, window_s=60.0)
    real = "198.51.100.7"
    for fake in ("1.1.1.1", "2.2.2.2", "3.3.3.3"):
        req = _FakeRequest({"X-Forwarded-For": f"{fake}, {real}"})
        allowed = lim.check(client_key(req), now=0.0)[0]
    # Third request from the same real peer is refused despite three
    # different forged prefixes.
    assert allowed is False


def test_malformed_forwarded_header_falls_back_to_socket():
    assert client_key(_FakeRequest({"X-Forwarded-For": " , , "}, peer="10.1.1.1")) == "10.1.1.1"

"""Per-client rate limiting for the public demo.

WHY THIS LIVES IN THE APP AND NOT IN CADDY
------------------------------------------
The obvious place for rate limiting is the reverse proxy, and the first
version of the Caddyfile did exactly that. It was wrong in a way that
only shows up at deploy time: Caddy's `rate_limit` directive comes from
a third-party module (caddy-ratelimit), not from the standard binary.
The `caddy:2-alpine` image does not have it, and Caddy refuses to start
on an unrecognised directive rather than ignoring it - so the whole
compose stack would have failed to come up.

The fix could have been a custom Caddy build via xcaddy. That pulls a Go
toolchain into the build and needs more memory than a small free-tier
instance comfortably has, to add one directive. Doing it in the
application instead costs ~40 lines, works behind any proxy or none,
and can be tested in CI like anything else.

The limit exists because the gateway runs a model on every allowed
request. Without it, a public demo is someone else's free inference
endpoint, and a free-tier quota disappears in minutes.

THREAT NOTE ON X-Forwarded-For
------------------------------
The client controls its own headers, so a request can arrive with
`X-Forwarded-For: 1.2.3.4` already set and a new fake value every time,
which would defeat per-IP limiting entirely. Caddy *appends* the real
peer address to whatever was already there, so the rightmost entry is
the one the proxy observed and the only one that cannot be forged.
Reading the leftmost entry - which is the common convention, and what
most snippets do - is the bug.

When no proxy is in front, the header is absent and the socket address
is used directly.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import Request

# FastAPI is imported lazily inside install(). The limiter logic itself
# has no web-framework dependency, so the test suite (and CI, which
# installs no server packages) can exercise the part that carries the
# security property without pulling in a server.


class SlidingWindowLimiter:
    """Fixed number of events per window, per key, in process memory.

    In-memory is the right scope here: one container serves the demo, and
    a limiter that loses its state on restart is a far smaller problem
    than a Redis dependency on a single-node deployment.
    """

    def __init__(self, events: int = 30, window_s: float = 60.0, max_keys: int = 10_000):
        self.events = events
        self.window_s = window_s
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic() if now is None else now
        q = self._hits.get(key)
        if q is None:
            # Bound the table. An attacker rotating source addresses would
            # otherwise turn the limiter itself into the memory leak it
            # exists to prevent.
            if len(self._hits) >= self.max_keys:
                self._evict(now)
            q = self._hits.setdefault(key, deque())

        cutoff = now - self.window_s
        while q and q[0] <= cutoff:
            q.popleft()

        if len(q) >= self.events:
            return False, max(0.0, q[0] + self.window_s - now)

        q.append(now)
        return True, 0.0

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_s
        for k in [k for k, v in self._hits.items() if not v or v[-1] <= cutoff]:
            self._hits.pop(k, None)
        if len(self._hits) >= self.max_keys:  # still full: drop oldest-seen
            for k in sorted(self._hits, key=lambda k: self._hits[k][-1])[: self.max_keys // 4]:
                self._hits.pop(k, None)


def client_key(request: Request) -> str:
    """Identify the caller, preferring the proxy-observed address.

    See the module docstring: the RIGHTMOST X-Forwarded-For entry is the
    one appended by the trusted proxy. Everything to its left was sent by
    the client and may be fabricated.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "unknown"


def install(app, events: int = 30, window_s: float = 60.0, path_prefix: str = "/v1/") -> None:
    """Attach the limiter to model-invoking routes only.

    /health and the static UI are deliberately exempt: they cost nothing
    to serve, and rate-limiting a health check breaks orchestration.
    """
    from fastapi import Request  # noqa: F811  (lazy: see module header)
    from fastapi.responses import JSONResponse

    limiter = SlidingWindowLimiter(events=events, window_s=window_s)
    app.state.rate_limiter = limiter

    @app.middleware("http")
    async def _rate_limit(request: Request, call_next):
        if not request.url.path.startswith(path_prefix):
            return await call_next(request)

        allowed, retry_after = limiter.check(client_key(request))
        if not allowed:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(max(1, int(retry_after) + 1))},
                content={
                    "detail": (
                        f"Rate limit exceeded: {events} requests per "
                        f"{int(window_s)}s. This demo runs a real model on "
                        "every request."
                    )
                },
            )
        return await call_next(request)

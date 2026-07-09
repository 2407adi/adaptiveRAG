"""The building's security: membership cards, doormen, and the tally counter.

Block 4.2. Cards (API keys) live in .env — never in YAML, which is committed
to git. The doorman is a FastAPI dependency (a function FastAPI runs BEFORE
the endpoint; if it raises, the endpoint never runs), attached once at the
router level so every service window gets him. /health lives on a separate,
doorman-free router — Azure's probe carries no card.
"""
import secrets
import time

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

# The card slot on every window: "show your X-API-Key header".
# auto_error=False → WE write the 401 message, not FastAPI's generic one.
# Side perk: /docs grows a padlock + Authorize button from this declaration.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """The tally counter: N visits per card per clock-minute (fixed window).

    In-memory dict → resets on restart, per-process only. Right-sized for
    one container; a multi-replica deployment would need a shared counter
    (e.g. Redis) instead.
    """

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._counts: dict[str, tuple[int, int]] = {}   # card → (minute, hits)

    def allow(self, key: str) -> bool:
        window = int(time.time() // 60)          # which clock-minute is it?
        seen_window, hits = self._counts.get(key, (window, 0))
        if seen_window != window:                # minute rolled over →
            hits = 0                             #   fresh page on the clicker
        if hits >= self.per_minute:
            return False                         # over budget: bounce (429)
        self._counts[key] = (window, hits + 1)   # tick, remember, admit
        return True


def require_api_key(request: Request, key: str | None = Security(api_key_header)) -> str:
    """The doorman: check the card, tick the tally, return the card's color (role)."""
    cfg = request.app.state.settings.auth
    if not cfg.enabled:
        return "admin"                           # doorman off duty (local dev): all staff

    if not key:
        raise HTTPException(status_code=401, detail="missing API key")   # no card

    role = None
    for known, known_role in request.app.state.api_keys.items():
        # compare_digest = constant-time compare: reads EVERY character even
        # after a mismatch, so response timing leaks nothing about the key.
        if secrets.compare_digest(key, known):
            role = known_role
    if role is None:
        raise HTTPException(status_code=401, detail="invalid API key")   # forged card

    if not request.app.state.rate_limiter.allow(key):
        raise HTTPException(status_code=429,                             # tally exceeded
                            detail="rate limit exceeded; try again next minute")
    return role


def require_role(required: str):
    """Factory for a stricter doorman: same card check, then a color check.

    require_api_key runs as a sub-dependency; FastAPI caches it per request,
    so the tally ticks once even when both doormen guard the same window.
    """
    def dep(role: str = Depends(require_api_key)) -> str:
        if role != required:
            raise HTTPException(status_code=403,                          # known card,
                                detail=f"requires {required} role")       # wrong color
        return role
    return dep

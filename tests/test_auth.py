"""Doorman drills: no card, forged card, wrong color, hammering, the dock scale,
and the archive ceiling. Real windows, fake staff — same pattern as test_api.py."""
from types import SimpleNamespace

from fastapi.testclient import TestClient

from adaptiverag.api.main import app
from adaptiverag.api.auth import RateLimiter
from adaptiverag.reason.router import QueryRoute

GOLD = "gold-key-for-tests"     # admin card
BLUE = "blue-key-for-tests"     # user card


def make_client(enabled=True, per_minute=10_000, max_upload_mb=1,
                max_total_chunks=100, store_count=0) -> TestClient:
    """Stage the building with adjustable house rules per drill."""
    app.state.settings = SimpleNamespace(auth=SimpleNamespace(
        enabled=enabled, rate_limit_per_minute=per_minute,
        max_upload_mb=max_upload_mb, max_total_chunks=max_total_chunks))
    app.state.api_keys = {GOLD: "admin", BLUE: "user"}
    app.state.rate_limiter = RateLimiter(per_minute)
    app.state.conversations = {}
    app.state.pipeline = SimpleNamespace(
        # minimal staff: every drill below routes DIRECT (no retrieval machinery)
        router=SimpleNamespace(classify=lambda q: SimpleNamespace(route=QueryRoute.DIRECT)),
        llm_client=SimpleNamespace(generate=lambda p: "direct answer"),
        vector_store=SimpleNamespace(count=lambda: store_count),   # dial the archive fill level
        ingest=SimpleNamespace(ingest=lambda d: {"files_processed": 1, "total_chunks": 1,
                                                 "corpus_summary": "test"}),
    )
    return TestClient(app)          # NO `with` → lifespan never runs; we staffed it by hand


def _q(client, key=None):
    headers = {"X-API-Key": key} if key else {}
    return client.post("/query", json={"question": "hello"}, headers=headers)


def _ingest(client, key, payload=b"hello"):
    return client.post("/ingest", files={"files": ("a.txt", payload, "text/plain")},
                       headers={"X-API-Key": key})


# ── card checks: 401 ────────────────────────────────────────────────────────

def test_no_card_gets_401():
    assert _q(make_client()).status_code == 401

def test_forged_card_gets_401():
    assert _q(make_client(), key="not-a-real-key").status_code == 401

def test_health_stays_open_to_the_street():
    # Azure's probe carries no card and must still see the "we're open" light.
    assert make_client().get("/health").status_code == 200

def test_doorman_off_duty_serves_everyone():
    assert _q(make_client(enabled=False)).status_code == 200


# ── card colors: 403 vs pass ────────────────────────────────────────────────

def test_blue_card_can_query():
    assert _q(make_client(), key=BLUE).status_code == 200

def test_blue_card_bounced_at_the_dock():
    # Known card, wrong color: 403 (forbidden), NOT 401 (unknown).
    assert _ingest(make_client(), BLUE).status_code == 403

def test_gold_card_can_query_and_ingest():
    client = make_client()
    assert _q(client, key=GOLD).status_code == 200
    assert _ingest(client, GOLD).status_code == 200


# ── the tally counter: 429 ──────────────────────────────────────────────────

def test_hammering_past_the_tally_gets_429():
    client = make_client(per_minute=3)
    for _ in range(3):
        assert _q(client, key=BLUE).status_code == 200   # within budget
    assert _q(client, key=BLUE).status_code == 429       # 4th knock bounced

def test_tally_is_per_card():
    client = make_client(per_minute=3)
    for _ in range(3):
        _q(client, key=BLUE)                             # BLUE spends its budget
    assert _q(client, key=GOLD).status_code == 200       # GOLD's clicker untouched


# ── the caps: 413 / 507 (apply to gold cards too — they go public in demos) ─

def test_oversized_package_gets_413():
    two_mb = b"x" * (2 * 1024 * 1024)                    # scale is set to 1 MB above
    assert _ingest(make_client(), GOLD, payload=two_mb).status_code == 413

def test_full_archive_refuses_even_gold_with_507():
    client = make_client(store_count=100, max_total_chunks=100)   # ceiling reached
    assert _ingest(client, GOLD).status_code == 507

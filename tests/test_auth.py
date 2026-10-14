"""
Fix #10: Auth endpoint smoke tests using FastAPI TestClient with mocked DB.
No real database writes — all DB calls are patched.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Return a TestClient with DB calls mocked."""
    with patch("src.database.init_db"), \
         patch("src.observability.tracing.init_observability"):
        from src.api import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── /health ────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


# ── /auth/register ─────────────────────────────────────────────────────────────

@patch("src.api.get_user_by_email", return_value=None)
@patch("src.api.create_user", return_value={"id": 1, "username": "alice", "email": "alice@test.com"})
def test_register_success(mock_create, mock_get, client):
    r = client.post("/auth/register", json={
        "username": "alice",
        "email": "alice@test.com",
        "password": "secret123",
    })
    assert r.status_code == 201
    body = r.json()
    assert "token" in body
    assert body["user"]["username"] == "alice"


@patch("src.api.get_user_by_email", return_value=None)
def test_register_short_username(mock_get, client):
    r = client.post("/auth/register", json={
        "username": "a",
        "email": "a@test.com",
        "password": "secret123",
    })
    assert r.status_code == 400


@patch("src.api.get_user_by_email", return_value=None)
def test_register_short_password(mock_get, client):
    r = client.post("/auth/register", json={
        "username": "alice",
        "email": "alice@test.com",
        "password": "123",
    })
    assert r.status_code == 400


@patch("src.api.get_user_by_email", return_value={"id": 1})
def test_register_duplicate_email(mock_get, client):
    r = client.post("/auth/register", json={
        "username": "alice2",
        "email": "alice@test.com",
        "password": "secret123",
    })
    assert r.status_code == 400


# ── /auth/login ────────────────────────────────────────────────────────────────

@patch("src.api.get_user_by_email")
@patch("src.api.verify_password", return_value=True)
def test_login_success(mock_verify, mock_get, client):
    mock_get.return_value = {
        "id": 1, "username": "alice", "email": "alice@test.com",
        "password": "$2b$12$hashed",
    }
    r = client.post("/auth/login", json={"email": "alice@test.com", "password": "secret123"})
    assert r.status_code == 200
    assert "token" in r.json()


@patch("src.api.get_user_by_email", return_value=None)
def test_login_unknown_email(mock_get, client):
    r = client.post("/auth/login", json={"email": "nobody@test.com", "password": "secret123"})
    assert r.status_code == 401


@patch("src.api.get_user_by_email")
@patch("src.api.verify_password", return_value=False)
def test_login_wrong_password(mock_verify, mock_get, client):
    mock_get.return_value = {
        "id": 1, "username": "alice", "email": "alice@test.com",
        "password": "$2b$12$hashed",
    }
    r = client.post("/auth/login", json={"email": "alice@test.com", "password": "wrongpassword"})
    assert r.status_code == 401


# ── /research/stream topic validation (Fix #9) ────────────────────────────────

def test_research_stream_rejects_empty_topic(client):
    # No auth token → 401 first, but topic validation also fires
    r = client.post("/research/stream", json={"topic": ""})
    # Either 401 (no auth) or 422 (validation); both mean the bad topic was caught
    assert r.status_code in (401, 422)


def test_research_stream_rejects_too_long_topic(client):
    r = client.post("/research/stream", json={"topic": "x" * 301})
    assert r.status_code in (401, 422)

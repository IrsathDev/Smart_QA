import os
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite:///./test_smartqa.db"
os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["AI_PROVIDER"] = "mock"
os.environ["SYNC_INDEXING"] = "true"
os.environ["STORAGE_ROOT"] = "./test_storage"
os.environ["APP_ENV"] = "test"

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database import engine


client = TestClient(app)


def setup_module() -> None:
    engine.dispose()
    Path("test_smartqa.db").unlink(missing_ok=True)


def csrf() -> dict[str, str]:
    return {"X-CSRF-Token": client.cookies.get("smartqa_csrf")}


def test_user_chat_and_isolation() -> None:
    with client:
        registered = client.post("/api/auth/register", json={"email": "owner@example.com", "password": "password123"})
        assert registered.status_code == 201
        conversation = client.post("/api/conversations", json={"title": "Private"}, headers=csrf())
        assert conversation.status_code == 201
        conversation_id = conversation.json()["id"]
        response = client.post(f"/api/conversations/{conversation_id}/messages:stream", json={"content": "Hello"}, headers=csrf())
        assert response.status_code == 200
        assert "response.completed" in response.text
        branches = client.get(f"/api/conversations/{conversation_id}/branches")
        assert branches.status_code == 200
        assert len(branches.json()) == 1
        unsupported = client.post("/api/documents", files={"files": ("image.png", b"nope", "image/png")}, headers=csrf())
        assert unsupported.status_code == 400
        client.post("/api/auth/logout", headers=csrf())
        client.post("/api/auth/register", json={"email": "other@example.com", "password": "password123"})
        assert client.get(f"/api/conversations/{conversation_id}").status_code == 404


def teardown_module() -> None:
    engine.dispose()
    Path("test_smartqa.db").unlink(missing_ok=True)

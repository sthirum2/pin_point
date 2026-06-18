from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_body() -> None:
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


def test_search_returns_501() -> None:
    response = client.get("/search", params={"q": "bicycle kick", "k": 3})
    assert response.status_code == 501


def test_search_default_k_returns_501() -> None:
    response = client.get("/search", params={"q": "header goal"})
    assert response.status_code == 501

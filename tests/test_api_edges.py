from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.inference import ModelLoadError
from app.main import create_app

MODEL_DIR = Path(__file__).parents[1] / "model"


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(model_dir=MODEL_DIR)) as test_client:
        yield test_client


@pytest.mark.parametrize("payload", [
    None,
    [],
    {"texts": "a string is not a list"},
    {"texts": [1]},
    {"texts": [None]},
    {"texts": ["valid", "\t\n"]},
    {"texts": ["x" * 10001]},
    {"texts": ["valid"] * 65},
])
def test_embeddings_validation_matrix(client, payload):
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("payload", [
    {},
    {"text_a": "valid"},
    {"text_a": "valid", "text_b": 5},
    {"text_a": "valid", "text_b": None},
    {"text_a": "valid", "text_b": "\n\t"},
    {"text_a": "valid", "text_b": "x" * 10001},
])
def test_similarity_validation_matrix(client, payload):
    assert client.post("/v1/similarity", json=payload).status_code == 422


def test_maximum_allowed_character_length_is_accepted(client):
    response = client.post("/v1/embeddings", json={"texts": ["x" * 10000]})
    assert response.status_code == 200
    assert len(response.json()["embeddings"][0]) == 384


def test_malformed_json_and_wrong_content_type_are_rejected(client):
    malformed = client.post("/v1/embeddings", content='{"texts": [', headers={"Content-Type": "application/json"})
    wrong_type = client.post("/v1/embeddings", content='{"texts": ["hello"]}', headers={"Content-Type": "text/plain"})
    assert malformed.status_code == 422
    assert wrong_type.status_code == 422


def test_not_ready_app_reports_not_ready_and_503():
    app = create_app(model_dir=MODEL_DIR)
    # TestClient without a lifespan context deliberately tests pre-startup state.
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "not_ready"
    assert client.post("/v1/embeddings", json={"texts": ["hello"]}).status_code == 503


def test_startup_failure_is_clear_for_missing_model_directory(tmp_path):
    with pytest.raises(ModelLoadError, match="must contain manifest.json"):
        with TestClient(create_app(model_dir=tmp_path)):
            pass


def test_inference_failure_returns_meaningful_http_error(client):
    class BrokenModel:
        def embed(self, texts):
            raise RuntimeError("internal runtime detail")

    original_model = client.app.state.model
    client.app.state.model = BrokenModel()
    try:
        response = client.post("/v1/embeddings", json={"texts": ["hello"]})
    finally:
        client.app.state.model = original_model
    assert response.status_code == 500
    assert response.json()["detail"] == "Embedding inference failed"


def test_similarity_identity_is_one_and_symmetric(client):
    same = client.post("/v1/similarity", json={"text_a": "A person playing golf", "text_b": "A person playing golf"})
    reverse = client.post("/v1/similarity", json={"text_a": "Someone swinging a golf club", "text_b": "A person playing golf"})
    forward = client.post("/v1/similarity", json={"text_a": "A person playing golf", "text_b": "Someone swinging a golf club"})
    assert same.status_code == reverse.status_code == forward.status_code == 200
    assert same.json()["similarity"] == pytest.approx(1.0, abs=1e-6)
    assert reverse.json()["similarity"] == pytest.approx(forward.json()["similarity"], abs=1e-6)
    assert -1.0 <= forward.json()["similarity"] <= 1.0


def test_unknown_route_is_not_found(client):
    assert client.get("/not-a-route").status_code == 404

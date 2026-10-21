import json
import os
from pathlib import Path

import numpy as np
import pytest
import onnxruntime as ort
from fastapi.testclient import TestClient
from transformers import AutoTokenizer

from app.inference import EmbeddingModel
from app.main import create_app

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parents[1] / "model"))


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(model_dir=MODEL_DIR)) as test_client:
        yield test_client


def test_model_loads_and_returns_normalized_384d_embeddings():
    model = EmbeddingModel(MODEL_DIR)
    vectors = model.embed(["A person playing golf", "Someone swinging a golf club"])
    assert vectors.shape == (2, 384)
    assert np.all(np.isfinite(vectors))
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embeddings_endpoint_returns_complete_vectors(client):
    response = client.post("/v1/embeddings", json={"texts": ["A person playing golf", "Someone swinging a golf club"]})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "social-embedding-v1.0"
    assert body["dimensions"] == 384
    assert len(body["embeddings"]) == 2
    assert all(len(vector) == 384 for vector in body["embeddings"])
    np.testing.assert_allclose(np.linalg.norm(body["embeddings"], axis=1), 1.0, atol=1e-5)


def test_similarity_is_cosine_of_normalized_embeddings(client):
    texts = ["A person playing golf", "Someone swinging a golf club"]
    vectors = client.post("/v1/embeddings", json={"texts": texts}).json()["embeddings"]
    expected = float(np.dot(vectors[0], vectors[1]))
    response = client.post("/v1/similarity", json={"text_a": texts[0], "text_b": texts[1]})
    assert response.status_code == 200
    assert response.json()["similarity"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("payload", [
    {"texts": []}, {"texts": [""]}, {"texts": ["   "]},
    {"texts": ["x" * 10001]}, {"texts": ["valid"] * 65},
])
def test_embeddings_reject_invalid_input(client, payload):
    assert client.post("/v1/embeddings", json=payload).status_code == 422


@pytest.mark.parametrize("payload", [
    {"text_a": "", "text_b": "valid"},
    {"text_a": "   ", "text_b": "valid"},
    {"text_a": "x" * 10001, "text_b": "valid"},
])
def test_similarity_rejects_invalid_input(client, payload):
    assert client.post("/v1/similarity", json=payload).status_code == 422


def test_embeddings_endpoint_supports_maximum_batch(client):
    response = client.post("/v1/embeddings", json={"texts": ["item {}".format(i) for i in range(64)]})
    assert response.status_code == 200
    assert len(response.json()["embeddings"]) == 64


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ready"
    assert body["model_ready"] is True


REFERENCE_PATH = Path(os.getenv("REFERENCE_EMBEDDINGS", Path(__file__).parent / "reference_embeddings.json"))


def test_quantized_onnx_matches_original_model_reference_embeddings():
    texts = ["A person playing golf", "Someone swinging a golf club"]
    if REFERENCE_PATH.is_file():
        reference = json.loads(REFERENCE_PATH.read_text())
        provenance = reference["provenance"]
        assert provenance["source"] == "original_finetuned_model"
        assert provenance["model"] == "social-embedding-v1.0"
        assert provenance["pooling"] == "cls"
        texts = reference["texts"]
        expected = np.asarray(reference["embeddings"], dtype=np.float32)
    elif (MODEL_DIR / "encoder_fp32.onnx").is_file():
        # The release's FP32 ONNX encoder is a genuine, locally supplied
        # reference implementation of the same fine-tuned model.
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="np")
        session = ort.InferenceSession(
            str(MODEL_DIR / "encoder_fp32.onnx"), providers=["CPUExecutionProvider"]
        )
        accepted = {item.name for item in session.get_inputs()}
        raw = session.run([session.get_outputs()[0].name], {k: v for k, v in tokens.items() if k in accepted})[0]
        pooled = np.asarray(raw[:, 0, :384], dtype=np.float32)
        expected = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    else:
        pytest.skip("No genuine references or release encoder_fp32.onnx were supplied")
    actual = EmbeddingModel(MODEL_DIR).embed(texts)
    assert expected.shape == actual.shape
    assert expected.shape[1] == 384
    assert np.all(np.isfinite(expected))
    assert np.all(np.linalg.norm(expected, axis=1) > 0.999)
    # Quantization can introduce vector drift while preserving embedding direction.
    cosine = np.sum(actual * expected, axis=1)
    assert np.all(cosine > 0.98)

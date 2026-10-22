import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from app.inference import EmbeddingModel, ModelLoadError

MODEL_DIR = Path(__file__).parents[1] / "model"


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel(MODEL_DIR)


def test_uses_cpu_execution_provider(model):
    assert model.session.get_providers()[0] == "CPUExecutionProvider"


def test_cls_pooling_truncation_and_l2_match_release_configuration(model):
    texts = ["short text", "a longer sentence with different padding behavior"]
    tokenized = model._tokenize(texts)
    raw = model.session.run(
        [model.output_name], {name: value for name, value in tokenized.items() if name in model.input_names}
    )[0]
    pooled = np.asarray(raw[:, 0, :384], dtype=np.float32)
    expected = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    np.testing.assert_allclose(model.embed(texts), expected, atol=1e-6, rtol=1e-6)


def test_embedding_is_invariant_to_batch_padding_within_quantization_tolerance(model):
    short = "A person playing golf"
    alone = model.embed([short])[0]
    in_batch = model.embed([short, "A much longer description " * 30])[0]
    # Dynamic quantization can shift activations slightly with sequence padding.
    assert float(np.dot(alone, in_batch)) > 0.99


def test_tokenization_truncates_to_release_maximum(model):
    encoded = model._tokenize(["golf " * 500])
    assert encoded["input_ids"].shape == (1, 128)
    assert encoded["attention_mask"].shape == (1, 128)


def test_compact_tokenizer_matches_original_transformers_token_ids(model):
    from transformers import AutoTokenizer

    texts = [
        "A person playing golf",
        "Someone swinging a golf club",
        "Café, dosa & idli!",
        "emoji 😊 and punctuation — yes",
        "A long input " * 80,
    ]
    reference = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)(
        texts, padding=True, truncation=True, max_length=128, return_tensors="np"
    )
    actual = model._tokenize(texts)
    np.testing.assert_array_equal(actual["input_ids"], reference["input_ids"])
    np.testing.assert_array_equal(actual["attention_mask"], reference["attention_mask"])
    np.testing.assert_array_equal(actual["token_type_ids"], reference["token_type_ids"])


def test_duplicate_inputs_are_deterministic_and_normalized(model):
    vectors = model.embed(["A person playing golf", "A person playing golf"])
    np.testing.assert_allclose(vectors[0], vectors[1], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_concurrent_embedding_calls_return_valid_vectors(model):
    texts = ["A person playing golf", "Someone swinging a golf club"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(lambda _: model.embed(texts), range(32)))
    for vectors in outputs:
        assert vectors.shape == (2, 384)
        assert np.all(np.isfinite(vectors))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("batch_size", [0, 65])
def test_model_rejects_out_of_range_batch_sizes(model, batch_size):
    with pytest.raises(ValueError, match="batch size"):
        model.embed(["text"] * batch_size)


def test_model_load_fails_clearly_when_release_files_are_missing(tmp_path):
    with pytest.raises(ModelLoadError, match="must contain manifest.json"):
        EmbeddingModel(tmp_path)


def test_model_load_rejects_non_cls_release_pooling(tmp_path):
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"max_sequence_length": 128}))
    (tmp_path / "1_Pooling" / "config.json").write_text(
        json.dumps({"embedding_dimension": 768, "pooling_mode": "mean"})
    )
    (tmp_path / "encoder_uint8.onnx").touch()
    with pytest.raises(ModelLoadError, match="unsupported pooling configuration"):
        EmbeddingModel(tmp_path)


def test_model_load_rejects_wrong_encoder_dimension(tmp_path):
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"max_sequence_length": 128}))
    (tmp_path / "1_Pooling" / "config.json").write_text(
        json.dumps({"embedding_dimension": 384, "pooling_mode": "cls"})
    )
    (tmp_path / "encoder_uint8.onnx").touch()
    with pytest.raises(ModelLoadError, match="expected a 768-dimensional"):
        EmbeddingModel(tmp_path)


def test_model_load_rejects_wrong_max_sequence_length(tmp_path):
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"max_sequence_length": 256}))
    (tmp_path / "1_Pooling" / "config.json").write_text(
        json.dumps({"embedding_dimension": 768, "pooling_mode": "cls"})
    )
    (tmp_path / "encoder_uint8.onnx").touch()
    with pytest.raises(ModelLoadError, match="maximum sequence length must be 128"):
        EmbeddingModel(tmp_path)

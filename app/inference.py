"""Local-only ONNX inference for the social embedding release."""

import json
import logging
import os
from pathlib import Path
import threading
from typing import Dict, List

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)
MODEL_NAME = "social-embedding-v1.0"
OUTPUT_DIMENSIONS = 384
MAX_SEQUENCE_LENGTH = 128
MAX_BATCH_SIZE = 64
MAX_TEXT_LENGTH = 10000


class ModelLoadError(RuntimeError):
    """The supplied local model release cannot be loaded safely."""


class EmbeddingModel:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._lock = threading.Lock()
        manifest_path = self.model_dir / "manifest.json"
        model_path = self.model_dir / "encoder_uint8.onnx"
        pooling_path = self.model_dir / "1_Pooling" / "config.json"
        if not manifest_path.is_file() or not model_path.is_file() or not pooling_path.is_file():
            raise ModelLoadError(
                "MODEL_DIR must contain manifest.json, encoder_uint8.onnx, and 1_Pooling/config.json"
            )
        try:
            manifest = json.loads(manifest_path.read_text())
            pooling = json.loads(pooling_path.read_text())
            if int(pooling.get("embedding_dimension", 0)) != 768:
                raise ValueError("expected a 768-dimensional encoder pooling configuration")
            if pooling.get("pooling_mode") != "cls":
                raise ValueError("unsupported pooling configuration; release specifies cls pooling")
            if int(manifest.get("max_sequence_length", 0)) != MAX_SEQUENCE_LENGTH:
                raise ValueError("release maximum sequence length must be 128")
            self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            pad_token = self.tokenizer.token_to_id("[PAD]")
            if pad_token is None:
                raise ValueError("tokenizer.json must define the [PAD] token")
            self.tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
            self.tokenizer.enable_padding(pad_id=pad_token, pad_token="[PAD]", pad_type_id=0)
            threads = max(1, int(os.getenv("ORT_INTRA_OP_THREADS", min(4, os.cpu_count() or 1))))
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            if self.session.get_providers()[0] != "CPUExecutionProvider":
                raise ValueError("ONNX Runtime did not activate CPUExecutionProvider")
            self.input_names = {item.name for item in self.session.get_inputs()}
            self.output_name = self.session.get_outputs()[0].name
            if "input_ids" not in self.input_names or "attention_mask" not in self.input_names:
                raise ValueError("ONNX model must accept input_ids and attention_mask")
            logger.info("model_loaded", extra={"model": MODEL_NAME, "provider": "CPUExecutionProvider"})
        except Exception as exc:
            raise ModelLoadError("Failed to load local model release from {}: {}".format(self.model_dir, exc)) from exc

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts or len(texts) > MAX_BATCH_SIZE:
            raise ValueError("batch size must be between 1 and 64")
        encoded = self._tokenize(texts)
        inputs = {key: value for key, value in encoded.items() if key in self.input_names}
        # ORT sessions are shared across requests; serialize run calls for safe CPU use.
        with self._lock:
            outputs = self.session.run([self.output_name], inputs)
        token_embeddings = np.asarray(outputs[0], dtype=np.float32)
        # The release uses Sentence Transformers CLS pooling. CLS selects token 0;
        # the attention mask is already applied by the transformer encoder itself.
        pooled = token_embeddings[:, 0, :]
        truncated = pooled[:, :OUTPUT_DIMENSIONS]
        norms = np.linalg.norm(truncated, ord=2, axis=1, keepdims=True)
        return truncated / np.maximum(norms, 1e-12)

    def _tokenize(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Encode texts with the bundled tokenizer.json; never accesses the network."""
        encodings = self.tokenizer.encode_batch(texts)
        return {
            "input_ids": np.asarray([encoding.ids for encoding in encodings], dtype=np.int64),
            "attention_mask": np.asarray([encoding.attention_mask for encoding in encodings], dtype=np.int64),
            "token_type_ids": np.asarray([encoding.type_ids for encoding in encodings], dtype=np.int64),
        }

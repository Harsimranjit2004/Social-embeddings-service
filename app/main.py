"""FastAPI application for CPU-only social text embeddings."""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.inference import (
    MAX_BATCH_SIZE,
    MAX_TEXT_LENGTH,
    MODEL_NAME,
    OUTPUT_DIMENSIONS,
    EmbeddingModel,
    ModelLoadError,
)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        item = {"level": record.levelname, "message": record.getMessage()}
        for key in ("model", "provider", "path", "status_code"):
            if hasattr(record, key):
                item[key] = getattr(record, key)
        if record.exc_info:
            item["exception"] = self.formatException(record.exc_info)
        return json.dumps(item)


_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), handlers=[_handler], force=True)
logger = logging.getLogger("embedding_service")


def _validate_text(value: str) -> str:
    if not value.strip():
        raise ValueError("text must not be empty or whitespace")
    if len(value) > MAX_TEXT_LENGTH:
        raise ValueError("text must be at most {} characters".format(MAX_TEXT_LENGTH))
    return value


class EmbeddingsRequest(BaseModel):
    texts: List[str] = Field(min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values):
        return [_validate_text(value) for value in values]


class SimilarityRequest(BaseModel):
    text_a: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    text_b: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text_a", "text_b")
    @classmethod
    def validate_text(cls, value):
        return _validate_text(value)


def create_app(model_dir: Optional[Path] = None) -> FastAPI:
    resolved_model_dir = Path(model_dir or os.getenv("MODEL_DIR", "/app/model"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model = EmbeddingModel(resolved_model_dir)
        except ModelLoadError:
            logger.exception("model_startup_failed", extra={"path": str(resolved_model_dir)})
            raise
        yield
        app.state.model = None

    app = FastAPI(title="Social Embedding Service", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        ready = getattr(request.app.state, "model", None) is not None
        return {"status": "ready" if ready else "not_ready", "model": MODEL_NAME, "model_ready": ready}

    @app.post("/v1/embeddings")
    def embeddings(payload: EmbeddingsRequest, request: Request):
        model = getattr(request.app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        try:
            vectors = model.embed(payload.texts)
        except Exception as exc:
            logger.exception("embedding_inference_failed")
            raise HTTPException(status_code=500, detail="Embedding inference failed") from exc
        return {
            "model": MODEL_NAME,
            "dimensions": OUTPUT_DIMENSIONS,
            "embeddings": vectors.tolist(),
        }

    @app.post("/v1/similarity")
    def similarity(payload: SimilarityRequest, request: Request):
        model = getattr(request.app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        try:
            vectors = model.embed([payload.text_a, payload.text_b])
        except Exception as exc:
            logger.exception("similarity_inference_failed")
            raise HTTPException(status_code=500, detail="Similarity inference failed") from exc
        score = float(np.dot(vectors[0], vectors[1]))
        return {"model": MODEL_NAME, "similarity": score}

    return app


app = create_app()

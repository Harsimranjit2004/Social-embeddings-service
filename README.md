# Social Embedding Service

A small CPU-only FastAPI service for the fine-tuned `social-embedding-v1.0` model. It uses the release's dynamically quantized UINT8 ONNX encoder, local tokenizer files, CLS pooling from `1_Pooling/config.json`, truncation to 384 dimensions, then L2 normalization. Model files are distributed separately from this source repository; startup only loads files already present in `MODEL_DIR` and never downloads model weights.

## Measured performance

| Metric | Result |
|---|---:|
| Single-text mean latency | **5.46 ms** |
| Throughput at two concurrent clients | **366.72 texts/s** |
| Peak sampled container memory | **294 MiB** |
| Optimized Docker image | **160.9 MiB** |
| Automated tests | **50 passed, 1 skipped** |

Measured on an Apple Silicon Docker host (`linux/arm64`) with a 2-CPU limit, two ONNX Runtime threads, and a 512 MiB memory cap. The service stayed healthy and returned zero errors during the reported load tests. These are local measurements, not AWS performance guarantees. The skipped test is original-model parity: genuine reference embeddings and the FP32 reference encoder were not included in the supplied release. See [BENCHMARK.md](BENCHMARK.md) for full latency distributions, stress results at both 512 MiB and 1 GiB, test details, and limits.

## Model files

Keep trained model artifacts on Hugging Face rather than committing them to GitHub. After publishing the release, download it into `model/` with the Hugging Face CLI. Set `HF_REPO_ID` to the actual model repository ID first:

```sh
python -m pip install -U huggingface_hub
export HF_REPO_ID=your-account/social-embedding-model-v1.0
hf download "$HF_REPO_ID" --repo-type model --local-dir model
```

The directory must include `encoder_uint8.onnx`, `manifest.json`, tokenizer files, `1_Pooling/config.json`, and `2_Normalize/config.json`. The service does not require `encoder_fp32.onnx` for production inference. Keep model files out of source control; `.gitignore` already excludes them. The supplied archive can also be unpacked locally:

```sh
unzip social_embedding_model_v1.zip -d model
```

Set `MODEL_DIR` to the release directory when it is elsewhere. The container defaults to two ONNX Runtime intra-op threads to match the recommended two-vCPU runtime profile; `ORT_INTRA_OP_THREADS` can override this for other CPU allocations.

## Run locally

Python 3.9 or newer is supported. Install the service dependencies and test tools, then launch:

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
MODEL_DIR=./model uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The process fails at startup with a clear error if required local model artifacts or a supported pooling configuration are missing.

## API

`GET /health` reports `ready` after the tokenizer and ONNX session load.

```sh
curl http://localhost:8000/health

curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"texts":["A person playing golf","Someone swinging a golf club"]}'

curl http://localhost:8000/v1/similarity \
  -H 'Content-Type: application/json' \
  -d '{"text_a":"A person playing golf","text_b":"Someone swinging a golf club"}'
```

Embedding requests accept 1–64 non-blank strings, each at most 10,000 characters. Tokenization truncates to 128 tokens. Invalid requests receive HTTP 422.

## Docker

Build from this directory after copying the model files into `model/`:

```sh
docker build -t social-embedding-service:1.0 .
docker run --rm -p 8000:8000 social-embedding-service:1.0
```

The image contains the selected UINT8 encoder and tokenizer/configuration artifacts (the unused FP32 encoder is excluded), runs as UID 10001, uses ONNX Runtime's `CPUExecutionProvider`, and includes a `/health` container health check. The production image omits Uvicorn's optional extras and Python bytecode; the measured ARM64 image is about 161 MiB. The recommended constrained profile is encoded in `compose.yaml`: 2 CPUs, 1 GiB RAM, no swap, and two ONNX Runtime threads.

```sh
docker compose up --build -d
docker compose ps
docker compose down
```

The same limits with `docker run` are `--cpus=2 --memory=1g --memory-swap=1g -e ORT_INTRA_OP_THREADS=2`. To run the 512 MiB no-swap profile through Compose, set `MEMORY_LIMIT=512m` for `docker compose up -d`; Compose uses that value for both the memory and memory-swap limit. For a smaller deployment image, a bind mount can supply a local model directory by overriding `MODEL_DIR` to the mounted path.

## Resource sizing measurements

On the available ARM64 Docker host (10 reported CPUs, 8.2 GB memory), both 2-CPU profiles—1 GiB and 512 MiB with swap disabled—loaded the model and passed concurrent request, batch, and long-input checks. At 512 MiB, the sampled memory peak was 294.1 MiB. At two concurrent clients, the service processed 183.36 requests/s (366.72 texts/s) with a 9.15 ms p50 and 19.24 ms p95. All stress requests succeeded with valid 384D response shapes. Results depend on the target CPU and request mix; see [BENCHMARK.md](BENCHMARK.md) for the full measurements and limitations.

## Tests and reference integration test

Run unit, real-model, API validation, model failure, concurrency, and normalization tests with:

```sh
pip install -r requirements-dev.txt
MODEL_DIR=./model python -m pytest -q
```

Run the Docker end-to-end tests against a constrained, non-root container (including the network-disabled startup check) with:

```sh
docker compose up --build -d
RUN_DOCKER_E2E=1 MODEL_DIR=./model python -m pytest tests/test_docker_e2e.py -v
```

The suite loads the actual ONNX release for model and API tests. The additional comparison test compares the UINT8 results against `tests/reference_embeddings.json` (or `REFERENCE_EMBEDDINGS`) when supplied. The fixture must contain `texts`, `embeddings`, and provenance identifying the original model and CLS pooling. Otherwise, if `encoder_fp32.onnx` is present in `model/`, the test generates its genuine reference embeddings locally using that release encoder. No reference vectors are invented. The supplied ZIP omitted both `encoder_fp32.onnx` and reference vectors, so this integration test skips until either source is available.

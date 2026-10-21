# Final container and performance measurements

Measured 2026-10-21 against the optimized `social-embedding-service:1.0` image and the local quantized model. These are measurements from the available Apple Silicon Docker host (`linux/arm64`, 10 reported host CPUs, 8.2 GB host memory), with the container capped at 2 CPUs and ONNX Runtime set to two intra-op threads. Request latency includes HTTP round-trip time from a local async client. Latency numbers are milliseconds.

## Artifact and runtime

- Image: 168,757,510 bytes (160.9 MiB), `linux/arm64`.
- Runtime user: UID/GID 10001.
- Runtime provider: `CPUExecutionProvider`.
- Limits: 2 CPUs, 512 MiB RAM, 512 MiB memory+swap (therefore no swap), two ONNX Runtime intra-op threads.
- Both 1 GiB and 512 MiB profiles started and loaded the model. Restart-to-ready was about 1.1 seconds. The first embedding call after a 1 GiB restart took 5.83 ms.
- Final container is running under the 512 MiB profile and reports healthy. It has had zero restarts and no OOM kill.

Run either limit with Compose:

```sh
# 1 GiB, no swap
docker compose up -d

# 512 MiB, no swap
 MEMORY_LIMIT=512m docker compose up -d
```

Both profiles keep the same two CPU limit and ONNX thread count.

## Warm request latency

Sequential requests after startup; each case measured independently. Percentiles use nearest-rank style over each small sample, so treat them as local reference numbers, not an SLA.

| Profile | Request case | Samples | Mean | p50 | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 GiB | 1 text | 100 | 6.46 | 5.50 | 11.77 | 19.81 | 32.84 |
| 1 GiB | 2 texts | 100 | 10.41 | 7.59 | 29.93 | 68.08 | 68.38 |
| 1 GiB | batch of 64 | 30 | 191.58 | 178.70 | 223.80 | 276.24 | 344.56 |
| 1 GiB | long text, near 128 tokens | 50 | 41.77 | 40.89 | 46.59 | 47.24 | 52.55 |
| 1 GiB | 10,000-character text (truncated to 128 tokens) | 20 | 47.88 | 46.30 | 53.99 | 53.99 | 67.66 |
| 1 GiB | similarity endpoint (two texts) | 100 | 7.27 | 6.94 | 8.80 | 10.54 | 12.27 |
| 512 MiB | 1 text | 100 | 5.46 | 5.19 | 6.95 | 8.13 | 20.82 |
| 512 MiB | 2 texts | 100 | 8.40 | 7.73 | 9.67 | 22.07 | 24.84 |
| 512 MiB | batch of 64 | 30 | 95.57 | 95.26 | 97.37 | 97.75 | 98.62 |
| 512 MiB | 10,000-character text (truncated to 128 tokens) | 20 | 42.22 | 42.21 | 42.93 | 42.93 | 43.02 |

Observed variation between runs is substantial for some cases (particularly the 64-item batch). The two profiles were measured on the same development host, not on an isolated production benchmark machine. Do not interpret the faster 512 MiB batch sample as evidence that a tighter memory cap improves speed.

## Sustained concurrency

Each row ran for 8 seconds with two short texts per request. Requests were checked for HTTP success and 384-value output shape. All reported error counts were zero.

| Memory | Concurrent clients | Requests | Requests/s | Texts/s | p50 ms | p95 ms | p99 ms | Max ms | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 GiB | 1 | 816 | 101.91 | 203.82 | 8.53 | 12.55 | 22.38 | 97.77 | 0 |
| 1 GiB | 2 | 1,404 | 175.37 | 350.75 | 9.61 | 19.56 | 22.71 | 71.74 | 0 |
| 1 GiB | 4 | 1,239 | 154.54 | 309.08 | 21.35 | 38.62 | 72.42 | 106.03 | 0 |
| 1 GiB | 8 | 1,419 | 176.58 | 353.15 | 42.38 | 57.84 | 72.99 | 101.30 | 0 |
| 512 MiB | 1 | 984 | 122.99 | 245.99 | 7.51 | 8.38 | 9.65 | 25.60 | 0 |
| 512 MiB | 2 | 1,468 | 183.36 | 366.72 | 9.15 | 19.24 | 21.80 | 42.59 | 0 |
| 512 MiB | 4 | 1,451 | 180.98 | 361.97 | 19.15 | 30.29 | 37.89 | 63.94 | 0 |
| 512 MiB | 8 | 1,440 | 179.00 | 358.00 | 41.58 | 52.84 | 63.97 | 107.50 | 0 |

Peak sampled container memory using Docker stats was about 285.2 MiB under the 1 GiB cap and 294.1 MiB under the 512 MiB cap. Peak sampled CPU was about 203% of one CPU, consistent with the 2-CPU quota. The 512 MiB run exercised all concurrency levels plus 64-item batches and a maximum-length input without OOM or endpoint errors. It leaves roughly 218 MiB between sampled peak and the hard cap on this host; additional model/runtime versions or deployment overhead may change that headroom.

## Output checks and automated tests

- Real API embeddings returned 384 values per text with measured L2 norms 1.00000002 and 0.999999997.
- Similarity endpoint returned 0.942073 for the golf sentence pair used in this run.
- Full local test command, including the opt-in Docker end-to-end tests: `RUN_DOCKER_E2E=1 MODEL_DIR=./model ../.venv/bin/python -m pytest -q` — **50 passed, 1 skipped**.
- The skipped test requires genuine reference vectors from the original fine-tuned model. Those vectors (and the FP32 ONNX artifact) are not in the supplied release directory, so an ONNX-vs-original numerical parity test remains unverified. No reference vectors were fabricated.
- Python compile check passed and `pip check` reported no broken requirements.

These results characterize this local ARM64 container. Repeat the benchmark on the intended production CPU and request mix before choosing an SLA or final capacity reservation.

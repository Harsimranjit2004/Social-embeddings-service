"""Opt-in end-to-end checks against the built, resource-limited Docker image.

Run with: RUN_DOCKER_E2E=1 python -m pytest tests/test_docker_e2e.py -v
"""

import json
import os
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pytest

IMAGE = "social-embedding-service:1.0"
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DOCKER_E2E") != "1",
    reason="set RUN_DOCKER_E2E=1 to start and test the Docker image",
)


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


@pytest.fixture(scope="module")
def service():
    try:
        docker("image", "inspect", IMAGE)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("Docker CLI or the built service image is unavailable")
    name = "social-embedding-e2e-" + uuid.uuid4().hex[:8]
    container_id = docker(
        "run", "-d", "--name", name,
        "--cpus=2", "--memory=1g", "--memory-swap=1g",
        "-e", "ORT_INTRA_OP_THREADS=2",
        "-p", "127.0.0.1::8000", IMAGE,
    )
    try:
        port_spec = docker("port", container_id, "8000/tcp").splitlines()[0]
        base_url = "http://127.0.0.1:" + port_spec.rsplit(":", 1)[1]
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                response = httpx.get(base_url + "/health", timeout=1)
                if response.status_code == 200 and response.json().get("model_ready"):
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            logs = docker("logs", container_id)
            pytest.fail("Docker service did not become ready within 60s:\n" + logs)
        with httpx.Client(base_url=base_url, timeout=30) as client:
            yield client, container_id
    finally:
        subprocess.run(["docker", "rm", "-f", container_id], check=False, capture_output=True)


def test_container_uses_expected_runtime_constraints_and_non_root_user(service):
    _, container_id = service
    settings = json.loads(docker("inspect", container_id))[0]
    assert settings["Config"]["User"] == "10001:10001"
    assert settings["HostConfig"]["NanoCpus"] == 2_000_000_000
    assert settings["HostConfig"]["Memory"] == 1024**3
    assert settings["HostConfig"]["MemorySwap"] == 1024**3
    assert "ORT_INTRA_OP_THREADS=2" in settings["Config"]["Env"]
    assert settings["State"]["OOMKilled"] is False


def test_container_health_embeddings_and_similarity(service):
    client, _ = service
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_ready"] is True

    texts = ["A person playing golf", "Someone swinging a golf club"]
    response = client.post("/v1/embeddings", json={"texts": texts})
    assert response.status_code == 200
    vectors = response.json()["embeddings"]
    assert len(vectors) == 2 and all(len(vector) == 384 for vector in vectors)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], atol=1e-5)

    similarity = client.post("/v1/similarity", json={"text_a": texts[0], "text_b": texts[1]})
    assert similarity.status_code == 200
    assert similarity.json()["similarity"] == pytest.approx(float(np.dot(*vectors)), abs=1e-6)


def test_container_handles_max_batch_and_concurrent_requests(service):
    client, container_id = service
    maximum_batch = client.post("/v1/embeddings", json={"texts": ["batch item"] * 64})
    assert maximum_batch.status_code == 200
    assert len(maximum_batch.json()["embeddings"]) == 64

    def request_embedding(_):
        with httpx.Client(base_url=str(client.base_url), timeout=30) as separate_client:
            return separate_client.post("/v1/embeddings", json={"texts": ["one", "two"]})

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(request_embedding, range(24)))
    assert all(response.status_code == 200 for response in responses)
    assert all(len(response.json()["embeddings"]) == 2 for response in responses)
    state = json.loads(docker("inspect", container_id))[0]["State"]
    assert state["Status"] == "running"
    assert state["OOMKilled"] is False


def test_container_starts_and_infers_without_network_access():
    name = "social-embedding-offline-" + uuid.uuid4().hex[:8]
    container_id = docker(
        "run", "-d", "--name", name, "--network=none",
        "--cpus=2", "--memory=1g", "--memory-swap=1g",
        "-e", "ORT_INTRA_OP_THREADS=2",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        IMAGE,
    )
    try:
        health_check = (
            "from urllib.request import urlopen; "
            "print(urlopen('http://127.0.0.1:8000/health', timeout=1).read().decode())"
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["docker", "exec", container_id, "python", "-c", health_check],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and '"model_ready":true' in result.stdout:
                break
            time.sleep(0.2)
        else:
            pytest.fail("offline container did not become ready:\n" + docker("logs", container_id))

        inference_check = (
            "import json; from urllib.request import Request, urlopen; "
            "body=json.dumps({'texts':['offline model inference']}).encode(); "
            "request=Request('http://127.0.0.1:8000/v1/embeddings', data=body, "
            "headers={'Content-Type':'application/json'}); "
            "print(json.dumps(json.loads(urlopen(request, timeout=10).read())))"
        )
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", inference_check],
            check=True,
            capture_output=True,
            text=True,
        )
        body = json.loads(result.stdout)
        assert body["dimensions"] == 384
        assert len(body["embeddings"]) == 1
        assert len(body["embeddings"][0]) == 384
    finally:
        subprocess.run(["docker", "rm", "-f", container_id], check=False, capture_output=True)

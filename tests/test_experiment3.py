import asyncio
import hashlib
import io
import json
import re
import subprocess
import sys

import pandas as pd
import pytest
from PIL import Image

httpx = pytest.importorskip("httpx")

from experiments.experiment2_remote_png import execute, parse_args
from experiments.experiment3_preprocessed_png import compare_experiment2
from src.client.http_client import InferenceHTTPClient
from src.dataset import ImageRecord
from src.image_geometry import PREPROCESSING_ID


def test_client_preprocessing_does_not_import_torch() -> None:
    subprocess.run([sys.executable, "-c", (
        "import experiments.experiment3_preprocessed_png; import sys; "
        "assert 'torch' not in sys.modules; assert 'torchvision' not in sys.modules"
    )], check=True)


def test_experiment3_has_independent_defaults(monkeypatch) -> None:
    monkeypatch.setenv("EXPERIMENT2_IMAGES", "10")
    args = parse_args(3, ["--server-url", "https://example.test", "--api-key", "test-key"])
    assert args.preprocess_on_client and args.experiment == 3
    assert args.num_images == 500 and args.runs == 5 and args.warmup == 20
    assert args.concurrencies == [1, 2, 4]
    assert not parse_args(2, ["--server-url", "https://example.test", "--api-key", "test-key"]).preprocess_on_client


def _write_run(root, *, byte_count=1000, seconds=10.0, image_id="img-1", network="test network", failed=False):
    run = root / "run_c1" / "run_01"
    run.mkdir(parents=True)
    metrics = {
        "configuration": {
            "num_images": 1, "concurrency": 1, "warmup": 2, "timeout_seconds": 60,
            "max_retries": 2, "network_conditions": network, "server_url_host": "example.test",
            "model": "efficientnet_b0", "weights": "IMAGENET1K_V1", "dataset_fingerprint": "same-source",
        },
        "total_seconds": seconds, "latency": {"p95_ms": seconds * 1000},
        "classification": {"accuracy_top1": 1.0},
    }
    (run / "metrics.json").write_text(json.dumps(metrics))
    pd.DataFrame([{
        "image_id": image_id, "predicted_class_id": 0,
        "png_bytes": byte_count, "status": "error" if failed else "ok",
    }]).to_csv(run / "predictions.csv", index=False)


def test_comparison_measures_actual_bytes_speedup_and_prediction_agreement(tmp_path) -> None:
    e2, e3 = tmp_path / "e2", tmp_path / "e3"
    _write_run(e2)
    _write_run(e3, byte_count=250, seconds=5.0)
    row = compare_experiment2(e3, e2).iloc[0]
    assert row["status"] == "available"
    assert row["png_reduction_percent"] == 75.0
    assert row["speedup_e2_over_e3"] == 2.0
    assert row["top1_match_rate"] == 1.0
    assert (e3 / "figures" / "13_comparison_experiment2.png").is_file()


@pytest.mark.parametrize("changes,expected", [
    ({"image_id": "different"}, "incompatible image IDs"),
    ({"network": "different"}, "incompatible: network_conditions"),
    ({"failed": True}, "incomplete"),
])
def test_comparison_refuses_mismatched_workloads(tmp_path, changes, expected) -> None:
    e2, e3 = tmp_path / "e2", tmp_path / "e3"
    _write_run(e2, **changes)
    _write_run(e3)
    row = compare_experiment2(e3, e2).iloc[0]
    assert row["status"].startswith(expected)
    assert "speedup_e2_over_e3" not in row


def test_comparison_allows_running_without_experiment2_results(tmp_path) -> None:
    _write_run(tmp_path / "e3")
    row = compare_experiment2(tmp_path / "e3", tmp_path / "missing").iloc[0]
    assert row["status"] == "experiment2 results unavailable"


def test_runner_writes_metrics_and_preserves_existing_results(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    records = []
    for index in range(2):
        source = data / f"image-{index}.png"
        Image.new("RGB", (321, 480), (20 + index, 50, 80)).save(source)
        records.append(ImageRecord(
            image_id=f"img-{index}", filename=source.name, relative_path=source.name,
            class_wnid="n01440764", class_name="tench", imagenet_class_id=0,
            width=321, height=480, original_size_bytes=source.stat().st_size,
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ))
    monkeypatch.setattr("experiments.experiment2_remote_png.load_manifest", lambda *args, **kwargs: records)
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "model_loaded": True, "preprocessing_id": PREPROCESSING_ID})
        assert request.url.path == "/predict/preprocessed"
        image_id = re.search(b'name="image_id"\r\n\r\n([^\r]+)', request.content).group(1).decode()
        png_start = request.content.index(b"\x89PNG\r\n\x1a\n")
        with Image.open(io.BytesIO(request.content[png_start:])) as uploaded:
            assert uploaded.size == (224, 224)
        return httpx.Response(200, json={
            "image_id": image_id, "model": "efficientnet_b0", "preprocessing_id": PREPROCESSING_ID,
            "predicted_class_id": 0, "predicted_class_name": "tench", "confidence_top1": 0.9,
            "top5_ids": [0, 1, 2, 3, 4], "top5_probabilities": [.9, .04, .03, .02, .01],
            "server_metrics": {"queue_wait_ms": .1, "decode_ms": .2, "preprocess_ms": .3,
                               "inference_ms": 1., "postprocess_ms": .1, "total_internal_ms": 1.7},
        })

    monkeypatch.setattr("experiments.experiment2_remote_png.InferenceHTTPClient", lambda *args, **kwargs: InferenceHTTPClient(
        *args, **kwargs, transport=httpx.MockTransport(handler),
    ))
    args = parse_args(3, [
        "--server-url", "https://example.test", "--api-key", "test-key", "--data-root", str(data),
        "--results-root", str(tmp_path / "e3"), "--num-images", "2", "--runs", "1",
        "--warmup", "1", "--concurrencies", "1", "--baseline-summary", str(tmp_path / "absent.csv"),
        "--baseline-predictions", str(tmp_path / "absent.csv"),
    ])
    asyncio.run(execute(args))
    assert calls.count("/predict/preprocessed") == 3  # one warmup, two measured requests
    run = args.results_root / "run_c1" / "run_01"
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["successful_images"] == 2
    assert metrics["configuration"]["spatial_preprocessing_location"] == "client"
    assert metrics["stage_latency"]["client_preprocess_ms"]["mean_ms"] > 0
    before = (run / "predictions.csv").read_bytes()
    with pytest.raises(FileExistsError):
        asyncio.run(execute(args))
    assert (run / "predictions.csv").read_bytes() == before

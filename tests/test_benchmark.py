"""Tests for benchmark helpers (percentiles, per-sample records, file hashing)."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from edge_vision.benchmark import file_sha256, file_size_mb


def test_percentile_ordering():
    latencies = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    assert p50 <= p95 <= p99


def test_per_sample_record_fields():
    record = {
        "sample_id": 0,
        "latency_ms": 5.5,
        "prediction": 3,
        "label": 3,
        "correct": True,
    }
    assert set(record) == {"sample_id", "latency_ms", "prediction", "label", "correct"}
    assert record["correct"] is True


def test_file_size_and_hash():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "file.bin")
        content = b"benchmark test content"
        with open(path, "wb") as f:
            f.write(content)
        size_mb = file_size_mb(path)
        assert abs(size_mb - (len(content) / (1024 * 1024))) < 1e-9
        digest = file_sha256(path)
        assert len(digest) == 64
        # Same content -> same hash
        assert file_sha256(path) == digest

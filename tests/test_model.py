"""Tests for the compact MobileNet-style model."""

from __future__ import annotations

import torch

from edge_vision.model import create_model


def test_output_shape_and_finite():
    model = create_model(num_classes=10)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 10), f"unexpected logits shape {logits.shape}"
    assert torch.isfinite(logits).all(), "logits must be finite"
    # Single-image deployment shape
    with torch.no_grad():
        single = model(torch.randn(1, 3, 32, 32))
    assert single.shape == (1, 10)


def test_num_classes_parameter():
    model = create_model(num_classes=5)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 5)

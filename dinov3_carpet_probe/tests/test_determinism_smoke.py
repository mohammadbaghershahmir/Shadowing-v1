"""Determinism smoke tests for PCA / clustering / anchors."""

from __future__ import annotations

import numpy as np

from dinov3_carpet_probe.src.clustering import cluster_features
from dinov3_carpet_probe.src.pca_visualizer import fit_transform_pca
from dinov3_carpet_probe.src.reproducibility import set_seed
from dinov3_carpet_probe.src.similarity import default_anchors, normalized_to_feature_index


def test_pca_deterministic():
    set_seed(0)
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(16, 20, 32)).astype(np.float32)
    a, _ = fit_transform_pca(feats, random_state=0)
    b, _ = fit_transform_pca(feats, random_state=0)
    assert np.allclose(a, b)


def test_kmeans_deterministic():
    set_seed(0)
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(12, 12, 8)).astype(np.float32)
    la, ma = cluster_features(feats, 4, random_state=0)
    lb, mb = cluster_features(feats, 4, random_state=0)
    assert np.array_equal(la, lb)
    assert ma["algorithm"] == "MiniBatchKMeans"
    assert all(c["n_components"] >= 0 for c in ma["clusters"])


def test_anchor_roundtrip():
    anchors = default_anchors()
    assert len(anchors) == 6
    for a in anchors:
        r, c = normalized_to_feature_index(a["u"], a["v"], (64, 64))
        assert 0 <= r < 64 and 0 <= c < 64

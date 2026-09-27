"""Tests for CSN-V4 scene splitting."""
from __future__ import annotations

import pytest

from csn_v4.data.splits import assign_scene_splits, assert_no_split_leakage


def _scene(i: int, *, group: str, sha: str | None = None) -> dict:
    h = sha or f"hash_{group}"
    return {
        "scene_id": f"scene_{i}",
        "group_id": group,
        "family_id": group,
        "design_id": group,
        "source_sha256": h,
        "source_bw_path": f"/fake/{i}/source_bw.bmp",
        "target_semantic_path": f"/fake/{i}/target_semantic.bmp",
        "artifact_dir": f"/fake/{i}",
        "full_artifact_dir": f"/fake/{i}/full",
        "image_width": 512,
        "image_height": 512,
    }


@pytest.mark.parametrize("n", [2, 3, 4])
def test_leave_one_out_has_train_and_val(n: int):
    scenes = [_scene(i, group=f"g{i}") for i in range(n)]
    split = assign_scene_splits(scenes, seed=42)
    assert split.policy == "leave_one_group_out"
    assert len(split.train_scenes) == n - 1
    assert len(split.val_scenes) == 1
    assert split.n_val_groups == 1
    assert split.n_train_groups == n - 1
    assert_no_split_leakage(split.train_scenes, split.val_scenes)


def test_same_sha256_different_groups_stay_together():
    scenes = [
        _scene(0, group="g0", sha="shared"),
        _scene(1, group="g1", sha="shared"),
    ]
    split = assign_scene_splits(scenes, seed=42)
    # Both must be in same split because sha256 links them
    train_ids = {s["scene_id"] for s in split.train_scenes}
    val_ids = {s["scene_id"] for s in split.val_scenes}
    assert ("scene_0" in train_ids) == ("scene_1" in train_ids)


def test_same_family_different_groups_stay_together():
    scenes = [
        _scene(0, group="g0"),
        _scene(1, group="g1"),
        _scene(2, group="g2"),
        _scene(3, group="g3"),
        _scene(4, group="g4"),
    ]
    for s in scenes:
        s["family_id"] = "family_A"
        s["design_id"] = f"design_{s['group_id']}"
    split = assign_scene_splits(scenes, seed=7)
    # Shared family_id forces a single split unit → no family leakage possible.
    assert split.policy == "intra_scene_spatial_only" or (
        {s["family_id"] for s in split.train_scenes} & {s["family_id"] for s in split.val_scenes} == set()
    )
    if split.val_scenes:
        assert_no_split_leakage(split.train_scenes, split.val_scenes)


def test_same_design_id_stay_together():
    scenes = [
        _scene(0, group="g0"),
        _scene(1, group="g1"),
        _scene(2, group="g2"),
    ]
    for s in scenes:
        s["design_id"] = "design_shared"
        s["family_id"] = s["group_id"]
    split = assign_scene_splits(scenes, seed=3)
    assert split.n_groups == 1
    assert len(split.val_scenes) == 0
    assert split.policy == "intra_scene_spatial_only"


def test_design_id_leakage_detected():
    train = [_scene(0, group="g0")]
    val = [_scene(1, group="g1")]
    train[0]["design_id"] = "d_shared"
    val[0]["design_id"] = "d_shared"
    train[0]["family_id"] = "f0"
    val[0]["family_id"] = "f1"
    with pytest.raises(AssertionError, match="design_id"):
        assert_no_split_leakage(train, val)


def test_single_group_all_train():
    scenes = [_scene(0, group="only")]
    split = assign_scene_splits(scenes, seed=42)
    assert split.policy == "intra_scene_spatial_only"
    assert len(split.train_scenes) == 1
    assert len(split.val_scenes) == 0

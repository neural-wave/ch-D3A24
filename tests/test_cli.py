"""Tests for the batch CLI seam (``zir-run``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from zir_image_enhancer import cli
from zir_image_enhancer.pipeline import StudioResult


class FakePipeline:
    """Stands in for the fal-backed pipeline: returns the first photo as the packshot."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def run(self, photos, name, roles=()) -> StudioResult:  # noqa: ANN001
        self.calls.append((name, len(photos)))
        return StudioResult(Path(photos[0]), Path(photos[0]), (), 0.0, 0.0)


def _write_image(path: Path, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)).save(path)


def _make_dataset(root: Path) -> None:
    _write_image(root / "dev" / "d001" / "input_1.jpg")
    rows = [{"garment_id": "d001", "view": "front", "inputs": ["dev/d001/input_1.jpg"]}]
    (root / "dev").mkdir(parents=True, exist_ok=True)
    (root / "dev" / "inputs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )


def test_resolve_dataset_root_prefers_explicit(tmp_path: Path) -> None:
    manifest = tmp_path / "m.jsonl"
    assert cli._resolve_dataset_root(manifest, str(tmp_path / "ds")) == tmp_path / "ds"


def test_resolve_dataset_root_uses_split_parent(tmp_path: Path) -> None:
    manifest = tmp_path / "dev" / "inputs.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("", encoding="utf-8")
    assert cli._resolve_dataset_root(manifest, None) == tmp_path


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _make_dataset(dataset)
    out = tmp_path / "submissions"
    code = cli.main(
        [
            "--manifest",
            str(dataset / "dev" / "inputs.jsonl"),
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--team",
            "zir",
            "--dry-run",
        ]
    )
    assert code == 0
    assert not list(out.rglob("*.png"))


def test_run_writes_submission_layout(tmp_path: Path) -> None:
    fake = FakePipeline()
    dataset = tmp_path / "ds"
    _make_dataset(dataset)
    out = tmp_path / "submissions"
    code = cli.main(
        [
            "--manifest",
            str(dataset / "dev" / "inputs.jsonl"),
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--team",
            "zir",
        ],
        pipeline=fake,
    )
    assert code == 0
    assert (out / "zir" / "d001" / "front.png").is_file()
    assert fake.calls == [("d001", 1)]


def test_unknown_model_is_refused(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _make_dataset(dataset)
    args = ["--manifest", str(dataset / "dev" / "inputs.jsonl"), "--out", str(tmp_path)]
    with pytest.raises(SystemExit):
        cli.main([*args, "--team", "zir", "--model", "nano-banana-9", "--dry-run"])

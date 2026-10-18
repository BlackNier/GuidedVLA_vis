"""Regression coverage for /gs local HPC mounts versus explicit GCS URIs."""

import pathlib
from types import SimpleNamespace

import pytest

from openpi.shared import download
from openpi.shared import normalize
from openpi.training import config


@pytest.mark.parametrize(
    ("directory", "expected"),
    [
        ("/gs/bs/work/checkpoint/assets", "/gs/bs/work/checkpoint/assets/ybwowen/libero"),
        (pathlib.Path("/gs/bs/work/checkpoint/assets"), "/gs/bs/work/checkpoint/assets/ybwowen/libero"),
        ("/net/work/checkpoint/assets", "/net/work/checkpoint/assets/ybwowen/libero"),
        ("gs://valid-bucket/assets", "gs://valid-bucket/assets/ybwowen/libero"),
    ],
)
def test_norm_asset_paths_do_not_reinterpret_local_mounts(monkeypatch, directory, expected):
    seen = []
    monkeypatch.setattr(download, "maybe_download", lambda path: seen.append(path) or pathlib.Path("/cached"))
    sentinel = {"normalization": "loaded"}
    monkeypatch.setattr(normalize, "load", lambda path: sentinel)
    factory = config.LeRobotLiberoDataConfig(
        repo_id="test", assets=config.AssetsConfig(assets_dir=directory, asset_id="ybwowen/libero")
    )
    result = factory.create_base_config(pathlib.Path("/fallback"), SimpleNamespace(model_type=config.ModelType.PI0))
    assert seen == [expected]
    assert result.norm_stats is sentinel


def test_default_assets_directory_stays_local(monkeypatch):
    seen = []
    monkeypatch.setattr(download, "maybe_download", lambda path: seen.append(path) or pathlib.Path("/cached"))
    monkeypatch.setattr(normalize, "load", lambda path: {})
    factory = config.LeRobotLiberoDataConfig(repo_id="test", assets=config.AssetsConfig(asset_id="ybwowen/libero"))
    factory.create_base_config(pathlib.Path("/gs/bs/work/assets"), SimpleNamespace(model_type=config.ModelType.PI0))
    assert seen == ["/gs/bs/work/assets/ybwowen/libero"]

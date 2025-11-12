from textwrap import dedent
import pytest
from app.train.config import load_config
from pydantic import ValidationError


def test_valid_yaml_loads(tmp_path):
    good_yaml = tmp_path / "config.yaml"
    good_yaml.write_text(dedent("""
        resize: [64, 64]
        normalize: true
        dtype: float32
        augment:
          flip_left_right: false
          flip_up_down: false
          random_crop:
        shuffle: true
        shuffle_buffer_size: 2048
        batch_size: 64
        drop_remainder: false
        prefetch: true
    """), encoding="utf-8")

    cfg = load_config(good_yaml)  # returns a DataConfig if your loader does validation
    assert cfg.resize == (64, 64)
    assert cfg.normalize is True
    assert cfg.batch_size == 64
    assert cfg.augment.flip_left_right is False


def test_flip_left_right_conflicts_with_resize(tmp_path):
    bad_yaml = tmp_path / "config.yaml"
    bad_yaml.write_text(dedent("""
        resize: [64, 64]
        normalize: true
        dtype: float32
        augment:
          flip_left_right: true
          flip_up_down: false
          random_crop: [128, 128]
        shuffle: true
        shuffle_buffer_size: 2048
        batch_size: 64
        drop_remainder: false
        prefetch: true
    """), encoding="utf-8")

    with pytest.raises((ValidationError, SystemExit)) as exc:
        _ = load_config(bad_yaml)

    # Optional: assert message contains your custom text
    assert "resize cannot be used" in str(exc.value)


def test_batch_size_must_be_positive(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("batch_size: 0\n", encoding="utf-8")
    with pytest.raises((ValidationError, SystemExit)):
        _ = load_config(bad)


def test_shuffle_buffer_gt_zero(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("shuffle_buffer_size: -1\n", encoding="utf-8")
    with pytest.raises((ValidationError, SystemExit)):
        _ = load_config(bad)
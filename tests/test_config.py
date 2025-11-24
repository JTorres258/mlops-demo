from textwrap import dedent
import pytest
from app.train.config import load_config
from pydantic import ValidationError


def test_valid_yaml_loads(tmp_path):
    good_yaml = tmp_path / "config.yaml"
    good_yaml.write_text(
        dedent(
            """
        data:
            input_size: [224, 224]
            normalize: true
            dtype: float32

        pipeline:
            cache: true
            shuffle: true
            shuffle_buffer_size: 128
            batch_size: 8
            drop_remainder: false
            prefetch: true

        augment:
            flip_left_right: true
            flip_up_down: false
            random_crop: [180, 180]
            brightness: null

        train:
            epochs: 5
            learning_rate: 0.001
            optimizer: adam        # adam | sgd | rmsprop
            loss: sparse_categorical_crossentropy
            metrics: ["accuracy"]
            seed: 42
            mixed_precision: false
            strategy: auto         # auto | single | mirrored | tpu

        model:
            architecture: EfficientNetV2B0
            weights: imagenet
            num_classes: 120
    """
        ),
        encoding="utf-8",
    )

    cfg = load_config(good_yaml)  # returns a DataConfig if your loader does validation
    assert cfg.data.input_size == (224, 224)
    assert cfg.data.normalize is True
    assert cfg.pipeline.batch_size == 8
    assert cfg.augment.flip_left_right is True


def test_flip_left_right_conflicts_with_resize(tmp_path):
    bad_yaml = tmp_path / "config.yaml"
    bad_yaml.write_text(
        dedent(
            """
        data:
            input_size: [224, 224]
            resize: [180,180]
            normalize: true
            dtype: float32

        pipeline:
            cache: true
            shuffle: true
            shuffle_buffer_size: 128
            batch_size: 8
            drop_remainder: false
            prefetch: true

        augment:
            flip_left_right: true
            flip_up_down: false
            random_crop: true
            brightness: 0.1

        train:
            epochs: 5
            learning_rate: 0.001
            optimizer: adam        # adam | sgd | rmsprop
            loss: sparse_categorical_crossentropy
            metrics: ["accuracy"]
            seed: 42
            mixed_precision: false
            strategy: auto         # auto | single | mirrored | tpu

        model:
            architecture: EfficientNetV2B0
            weights: imagenet
            num_classes: 120
    """
        ),
        encoding="utf-8",
    )

    with pytest.raises((ValidationError, SystemExit)) as exc:
        _ = load_config(bad_yaml)

    # Optional: assert message contains your custom text
    assert "cropping and resizing" in str(exc.value.lower())


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

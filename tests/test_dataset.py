import tensorflow as tf

from app.train.dataset import get_dataset


def test_get_dataset():
    (ds_train, ds_test), ds_info = get_dataset(split=["train[:2]", "test[:2]"])

    assert ds_info.features["image"].shape == (None, None, 3)
    assert ds_info.features["label"].num_classes == 120
    assert ds_info.splits["train"].num_examples == 12000
    assert ds_info.splits["test"].num_examples == 8580

    assert isinstance(ds_train, tf.data.Dataset)
    assert isinstance(ds_test, tf.data.Dataset)

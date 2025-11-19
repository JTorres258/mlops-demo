import tensorflow as tf
import tensorflow_datasets as tfds
import os
from functools import partial
from app.train.config import load_config, ExperimentConfig

tf.keras.utils.set_random_seed(42)


def get_dataset(split="train"):
    """Download and load a TensorFlow dataset.
    Args:
        split: The dataset split to load ('train', 'test').

    Returns:
        A TensorFlow dataset object.
    """

    # Load the dataset
    ds, ds_info = tfds.load(
        "stanford_dogs",
        split=split,
        as_supervised=True,
        with_info=True,
        shuffle_files=False,
    )

    return ds, ds_info


def prepare_dataset(
    dataset: tf.data.Dataset,
    config: ExperimentConfig,
    *,
    training: bool,
    deterministic: bool | None = None,
) -> tf.data.Dataset:
    """
    Config-driven tf.data pipeline that applies your data_preprocessing function.
    Order: map -> cache -> (shuffle if train) -> batch -> prefetch.
    Args:
        dataset: A TensorFlow dataset to preprocess.
        config: A ExperimentConfig object containing preprocessing parameters.
        training: Whether the dataset is for training (enables shuffling).
        deterministic: If set, controls the determinism of the dataset pipeline.
    Returns:
        A preprocessed TensorFlow dataset.
    """

    # Optional determinism control (throughput vs. reproducibility)
    if deterministic is not None:
        opts = tf.data.Options()
        opts.experimental_deterministic = bool(deterministic)
        dataset = dataset.with_options(opts)

    # Preprocessing mapping
    map_fn = partial(data_preprocessing, config=config, training=training)
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Retrieve pipeline config
    pipe = config.pipeline

    # Cache if enabled (great for small/medium datasets or repeated epochs)
    if pipe.cache:
        dataset = dataset.cache()

    # Shuffle only for training
    if training and pipe.shuffle:
        dataset = dataset.shuffle(buffer_size=pipe.shuffle_buffer_size)

    # Batch (drop_remainder if you need static shapes)
    dataset = dataset.batch(pipe.batch_size, drop_remainder=pipe.drop_remainder)

    # Prefetch to overlap CPU prep & device compute
    if pipe.prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def data_preprocessing(
    image: tf.Tensor, label: tf.int64, *, config: ExperimentConfig, training: bool
):
    """Data preprocessing function applying transformations based on config.
    Args:
        image: Input image tensor.
        label: Corresponding label tensor.
        config: ExperimentConfig object with preprocessing parameters.
        training: Boolean indicating if in training mode (enables augmentations).
    Returns:
        Preprocessed image and label tensors.
    """

    # Ensure image has rank 3
    image = tf.ensure_shape(image, [None, None, None])  # helps shape inference

    # Retrieve data config
    dat = config.data

    # Resize video to the input size
    if dat.resize:
        image = tf.image.resize(image, dat.resize, method="bilinear")
    else:
        # Resize so that the shortest side = 256, then center crop to target_size.
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]

        # scale factor so min(h, w) → RESIZE_MIN
        scale = 256 / tf.cast(tf.minimum(h, w), tf.float32)
        new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)

        image = tf.image.resize(image, (new_h, new_w), method="bilinear")

    # Normalization: [0, 255] => [-1, 1] floats
    if dat.normalize:
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        if dat.dtype == "float16":
            image = tf.cast(image, tf.float16)

    # Augmentation
    if training and config.augment:
        # Retrieve augment config
        aug = config.augment

        # While deifining an if statement per augmentation is enough, here
        # we use a more defensive style that provides a default value if some
        # attributes are missing in the config object. Just as an example.

        if getattr(aug, "brightness", 0.1):
            image = tf.image.random_brightness(image, aug.brightness)

        if getattr(aug, "flip_left_right", False):
            image = tf.image.random_flip_left_right(image)

        if getattr(aug, "flip_up_down", False):
            image = tf.image.random_flip_up_down(image)

        if getattr(aug, "random_crop", False):
            # Check crop size is smaller than image size
            # This is done implicitly by tf.image.random_crop, but you can do:
            # tf.debugging.assert_less(
            #     tf.shape(crop_size)[:2], tf.shape(image)[:2],
            #     message="Crop size must be smaller than original image shape"
            # )

            image = tf.image.random_crop(
                image, size=[dat.input_size[0], dat.input_size[1], tf.shape(image)[-1]]
            )

    else:
        image = tf.image.resize_with_crop_or_pad(
            image, dat.input_size[0], dat.input_size[1]
        )

    # Labels usually don't need modification
    label_encoded = tf.one_hot(label, config.model.num_classes)

    return image, label_encoded


def save_jpg(tensor, filename):
    """
    Saves a 3D image tensor (H, W, C) in [0,1] float32 or uint8 as a JPG using Pillow.
    """
    # Convert to uint8 (Pillow expects values 0–255)
    if tensor.dtype != tf.uint8:
        tensor = tf.cast(tf.clip_by_value(tensor * 255.0, 0, 255), tf.uint8)

    # Convert to numpy
    arr = tensor.numpy()

    # Create and save with PIL
    img = Image.fromarray(arr)
    img.save(filename, format="JPEG")


if __name__ == "__main__":

    from PIL import Image

    # Example usage
    config = load_config("./configs/base_config.yml")
    train_ds, ds_info = get_dataset(split="train")

    setattr(config.pipeline, "shuffle", False)
    train_ds_prepared = prepare_dataset(train_ds, config, training=False)
    for batch in train_ds_prepared.take(1):
        images, labels = batch
        print(tf.math.reduce_max(images), tf.math.reduce_min(images))
        print(images.shape, labels.shape)
        print(labels)

    
    list_breed = []
    with open("/home/vscode/tensorflow_datasets/stanford_dogs/0.2.0/label.labels.txt", "r") as f:
        for line in f:
            breed = line.strip().split("-", 1)[1]  # split only once
            list_breed.append(breed)

    for raw_img, raw_label in train_ds.take(1):
        print(raw_img.shape)
        save_jpg(raw_img, f"original_{list_breed[raw_label]}.jpg")
    for aug_img, aug_label in train_ds_prepared.take(1):
        print(aug_img[0].shape)
        aug_label = tf.argmax(aug_label, axis=-1)
        print(aug_label[0])
        save_jpg(aug_img[0], f"augment_{list_breed[aug_label[0]]}.jpg")
    
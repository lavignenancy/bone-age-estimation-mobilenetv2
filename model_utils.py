import json
import os

import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = 224
COLOR_MODE = "RGB"
TARGET_SCALE_MONTHS = 240.0
RANDOM_SEED = 42

CONFIG_FILENAME = "preprocessing_config.json"


def load_and_preprocess_image(image_source, img_size=IMG_SIZE):
    if isinstance(image_source, str):
        img = Image.open(image_source)
    else:
        img = image_source

    img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr


def months_to_scaled(months):
    return np.asarray(months, dtype="float32") / TARGET_SCALE_MONTHS


def scaled_to_months(scaled_value):
    return np.asarray(scaled_value, dtype="float32") * TARGET_SCALE_MONTHS


def build_model(img_size=IMG_SIZE, freeze_backbone=True):
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = not freeze_backbone

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = backbone(inputs, training=False)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, "linear")(x)

    model = tf.keras.Model(inputs, outputs, name="bone_age_regressor")
    return model


def make_dataset(df, data_root, img_size=IMG_SIZE, shuffle=False, batch_size=32):
    paths = [os.path.join(data_root, f) for f in df["filename_id"]]
    labels = months_to_scaled(df["boneage"].to_numpy())

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, y: (tf.numpy_function(
            lambda p_: load_and_preprocess_image(p_.decode("utf-8"), img_size),
            [p], tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(lambda x, y: (tf.ensure_shape(x, (img_size, img_size, 3)), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, seed=RANDOM_SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def save_preprocessing_config(save_dir, extra_info=None):
    config = {
        "img_size": IMG_SIZE,
        "color_mode": COLOR_MODE,
        "target_scale_months": TARGET_SCALE_MONTHS,
        "random_seed": RANDOM_SEED,
        "output_unit": "months",
        "uses_sex_input": False,
        "preprocessing_function": "tf.keras.applications.mobilenet_v2.preprocess_input",
        "tensorflow_version": tf.__version__,
    }
    if extra_info:
        config.update(extra_info)

    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, CONFIG_FILENAME)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def load_preprocessing_config(save_dir):
    config_path = os.path.join(save_dir, CONFIG_FILENAME)
    with open(config_path, "r") as f:
        return json.load(f)

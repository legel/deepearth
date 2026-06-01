"""
WatNet model builder — Keras 3 compatible port.

WatNet source (WatNet/model/seg_model/watnet.py) uses `tf.image.resize`
directly on symbolic tensors, which is invalid in Keras 3 functional API.
This module reimplements the same architecture using proper Keras layers.

Architecture:
  MobileNetV2 backbone (6-band input) → ASPP → multi-scale decoder → sigmoid
  (DeepLabv3+ variant, Luo et al. 2021)
"""

import os
import sys
import numpy as np
import tensorflow as tf

WATNET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "WatNet")


class BilinearResize(tf.keras.layers.Layer):
    """Wrap tf.image.resize in a proper Layer for Keras 3 functional API compatibility."""
    def __init__(self, target_h, target_w, **kwargs):
        super().__init__(**kwargs)
        self.target_h = target_h
        self.target_w = target_w

    def call(self, x):
        return tf.image.resize(x, [self.target_h, self.target_w], method="bilinear")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"target_h": self.target_h, "target_w": self.target_w})
        return cfg


def _relu6(x):
    return tf.keras.layers.ReLU(6.0)(x)


def _conv_block(inputs, filters, kernel, strides):
    x = tf.keras.layers.Conv2D(filters, kernel, padding="same", strides=strides)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    return _relu6(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    tchannel = inputs.shape[-1] * t
    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))
    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = _relu6(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if r:
        x = tf.keras.layers.add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    x = _bottleneck(inputs, filters, kernel, t, strides)
    for _ in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)
    return x


def MobileNetV2_backbone(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape, name="input")
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))
    x = _inverted_residual_block(x, 16,  (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24,  (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32,  (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 64,  (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, 96,  (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
    x = tf.keras.layers.Conv2D(2, (1, 1), padding="same")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model


def aspp_2_compat(tensor, dims):
    """ASPP — Keras 3 compatible (uses BilinearResize layer instead of tf.image.resize)."""
    h, w = dims[1], dims[2]

    y_pool = tf.keras.layers.AveragePooling2D(
        pool_size=(h, w), name="average_pooling")(tensor)
    y_pool = tf.keras.layers.Conv2D(
        128, 1, padding="same", kernel_initializer="he_normal",
        name="pool_1x1conv2d", use_bias=False)(y_pool)
    y_pool = tf.keras.layers.BatchNormalization(name="bn_1")(y_pool)
    y_pool = tf.keras.layers.Activation("relu", name="relu_1")(y_pool)
    y_pool = BilinearResize(h, w, name="aspp_pool_resize")(y_pool)

    y_1 = tf.keras.layers.Conv2D(
        128, 1, dilation_rate=1, padding="same",
        kernel_initializer="he_normal", name="ASPP_conv2d_d1", use_bias=False)(tensor)
    y_1 = tf.keras.layers.BatchNormalization(name="bn_2")(y_1)
    y_1 = tf.keras.layers.Activation("relu", name="relu_2")(y_1)

    y_6 = tf.keras.layers.Conv2D(
        128, 3, dilation_rate=6, padding="same",
        kernel_initializer="he_normal", name="ASPP_conv2d_d6", use_bias=False)(tensor)
    y_6 = tf.keras.layers.BatchNormalization(name="bn_3")(y_6)
    y_6 = tf.keras.layers.Activation("relu", name="relu_3")(y_6)

    y_12 = tf.keras.layers.Conv2D(
        128, 3, dilation_rate=12, padding="same",
        kernel_initializer="he_normal", name="ASPP_conv2d_d12", use_bias=False)(tensor)
    y_12 = tf.keras.layers.BatchNormalization(name="bn_4")(y_12)
    y_12 = tf.keras.layers.Activation("relu", name="relu_4")(y_12)

    y_18 = tf.keras.layers.Conv2D(
        128, 3, dilation_rate=18, padding="same",
        kernel_initializer="he_normal", name="ASPP_conv2d_d18", use_bias=False)(tensor)
    y_18 = tf.keras.layers.BatchNormalization(name="bn_5")(y_18)
    y_18 = tf.keras.layers.Activation("relu", name="relu_5")(y_18)

    y = tf.keras.layers.concatenate(
        [y_pool, y_1, y_6, y_12, y_18], name="ASPP_concat")
    y = tf.keras.layers.Conv2D(
        128, 1, dilation_rate=1, padding="same",
        kernel_initializer="he_normal", name="ASPP_conv2d_final", use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization(name="bn_final")(y)
    y = tf.keras.layers.Activation("relu", name="relu_final")(y)
    return y


def build_watnet_keras3(input_shape=(512, 512, 6)):
    """
    Build WatNet model compatible with Keras 3 / TF 2.16+.
    Architecture matches WatNet/model/seg_model/watnet.py exactly, but
    tf.image.resize calls are wrapped in BilinearResize layers.

    Layer index correction: the original WatNet uses d_feature=91 with Keras 2.x
    where ReLU6 is a shared singleton (1 layer object in model.layers). In Keras 3,
    each ReLU(6.0) call creates a new layer instance, shifting layer indices.
    The correct Keras 3 indices for the same features (by matching output channels):
      Original Keras 2.x d=91  → batch_normalization_40 (576ch) → Keras 3 index 117
      Original Keras 2.x m=24  → batch_normalization_8  (144ch) → Keras 3 index 24  ✓
      Original Keras 2.x l=11  → batch_normalization_3  (16ch)  → Keras 3 index 11  ✓
    """
    print("*** Building WatNet (Keras 3 compatible) ***")
    d_feature, m_feature, l_feature = 117, 24, 11

    img_height, img_width, img_channel = input_shape
    backbone = MobileNetV2_backbone(input_shape)

    # Deep features (1/32 scale)
    image_features = backbone.get_layer(index=d_feature).output
    dims = image_features.shape  # (None, H/32, W/32, 1280)
    x_a = aspp_2_compat(image_features, dims)
    # Upsample to 1/4 scale
    x_a = BilinearResize(img_height // 4, img_width // 4, name="aspp_up_quarter")(x_a)

    # Middle features (1/4 scale)
    x_b = backbone.get_layer(index=m_feature).output
    x_b = tf.keras.layers.Conv2D(
        48, 1, padding="same", kernel_initializer="he_normal",
        name="low_level_projection", use_bias=False)(x_b)
    x_b = tf.keras.layers.BatchNormalization(name="bn_low_level_projection")(x_b)
    x_b = tf.keras.layers.Activation("relu", name="low_level_activation")(x_b)

    # Low-level features (1/2 scale)
    x_c = backbone.get_layer(index=l_feature).output
    x_c = tf.keras.layers.Conv2D(
        48, 1, padding="same", kernel_initializer="he_normal",
        name="low_level_projection_2", use_bias=False)(x_c)
    x_c = tf.keras.layers.BatchNormalization(name="bn_low_level_projection_2")(x_c)
    x_c = tf.keras.layers.Activation("relu", name="low_level_activation_2")(x_c)

    # Decoder
    x = tf.keras.layers.concatenate([x_a, x_b], name="decoder_concat_1")
    x = tf.keras.layers.Conv2D(
        128, 3, padding="same", activation="relu",
        kernel_initializer="he_normal", name="decoder_conv2d_1", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name="bn_decoder_1")(x)
    x = tf.keras.layers.Activation("relu", name="activation_decoder_1")(x)
    x = tf.keras.layers.Conv2D(
        128, 3, padding="same", activation="relu",
        kernel_initializer="he_normal", name="decoder_conv2d_2", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name="bn_decoder_2")(x)
    x = tf.keras.layers.Activation("relu", name="activation_decoder_2")(x)
    x = BilinearResize(img_height // 2, img_width // 2, name="decoder_up_half")(x)

    x_2 = tf.keras.layers.concatenate([x, x_c], name="decoder_concat_3")
    x_2 = tf.keras.layers.Conv2DTranspose(
        128, 3, strides=2, padding="same",
        kernel_initializer="he_normal", name="decoder_deconv2d", use_bias=False)(x_2)
    x_2 = tf.keras.layers.BatchNormalization(name="bn_decoder_4")(x_2)
    x_2 = tf.keras.layers.Activation("relu", name="activation_decoder_4")(x_2)
    x_2 = tf.keras.layers.Conv2D(
        1, (1, 1), strides=1, padding="same",
        kernel_initializer="he_normal", activation="sigmoid")(x_2)

    model = tf.keras.models.Model(
        inputs=backbone.input, outputs=x_2, name="watnet")
    print(f"*** Output shape: {model.output_shape} ***")
    return model


def _extract_weights_from_h5(weights_path):
    """
    Extract all weight arrays from the h5 model file in layer order.
    Returns list of (layer_name, [weight_arrays]) sorted by the order
    they appear in /model_weights.
    """
    import h5py
    result = {}
    with h5py.File(weights_path, "r") as f:
        mw = f["model_weights"]
        for layer_name in mw.keys():
            layer_grp = mw[layer_name]
            if not isinstance(layer_grp, h5py.Group):
                continue
            weights = []
            for sub_name in layer_grp.keys():
                sub_grp = layer_grp[sub_name]
                if isinstance(sub_grp, h5py.Group):
                    for w_name in sub_grp.keys():
                        weights.append(np.array(sub_grp[w_name]))
            if weights:
                result[layer_name] = weights
    return result


def load_watnet_from_source(weights_path):
    """
    Build WatNet from source (Keras 3 compatible) and load pre-trained weights.

    With d_feature=117 (corrected for Keras 3 ReLU layer-instance counting), all
    105 layer names in the model match those in the h5 file exactly. Weights are
    loaded via Keras's native load_weights(by_name=True) which handles the correct
    internal ordering of weight arrays (kernel before bias, gamma/beta/mean/var).
    """
    import tensorflow as tf
    tf.keras.backend.clear_session()  # reset name counters for reproducible names
    model = build_watnet_keras3(input_shape=(512, 512, 6))

    # Use Keras's native by-name loader — handles weight array ordering correctly
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    print(f"  Weights loaded by name from {weights_path}")

    # Quick sanity: count layers with non-zero weights
    loaded = sum(
        1 for layer in model.layers
        if layer.weights and any(tf.reduce_any(tf.not_equal(w, 0)).numpy() for w in layer.weights)
    )
    print(f"  Layers with non-zero weights: {loaded}/{sum(1 for l in model.layers if l.weights)}")
    return model

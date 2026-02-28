import tensorflow as tf
from tensorflow.keras import layers, Model


def build_projection_head(encoder: Model, projection_dim: int = 128) -> Model:
    inputs = encoder.input
    h = encoder.output
    z = layers.Dense(projection_dim, activation="relu", name="proj_dense")(h)
    z = layers.LayerNormalization(name="proj_norm")(z)
    z = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="proj_l2")(z)
    return Model(inputs=inputs, outputs=[h, z], name="encoder_with_projection")

from tensorflow.keras import layers, Model


def build_shared_encoder(input_dim: int = 1024, feature_dim: int = 256) -> Model:
    inputs = layers.Input(shape=(input_dim,), name="embedding_input")
    x = layers.Dense(512, activation="relu", name="enc_dense_512")(inputs)
    x = layers.BatchNormalization(name="enc_bn")(x)
    x = layers.Dropout(0.3, name="enc_dropout")(x)
    x = layers.Dense(feature_dim, activation="relu", name="enc_dense_features")(x)
    features = layers.LayerNormalization(name="features")(x)
    model = Model(inputs=inputs, outputs=features, name="shared_encoder")
    return model

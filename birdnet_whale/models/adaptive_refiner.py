import tensorflow as tf
from tensorflow.keras import layers, Model


class SelectiveGate(layers.Layer):
    def __init__(
        self,
        temperature_init: float = 1.5,
        target_mean: float = 0.45,
        balance_weight: float = 1e-3,
        min_std: float = 0.08,
        std_weight: float = 5e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature_init = float(max(1e-3, temperature_init))
        self.target_mean = float(min(max(target_mean, 0.05), 0.95))
        self.balance_weight = float(max(0.0, balance_weight))
        self.min_std = float(max(0.0, min_std))
        self.std_weight = float(max(0.0, std_weight))
        self._temp_raw = None
        self._last_gate_mean = None
        self._last_gate_std = None

    def build(self, input_shape):
        temp_raw_init = tf.math.log(tf.math.expm1(tf.constant(self.temperature_init, dtype=tf.float32)))
        self._temp_raw = self.add_weight(
            name="gate_temp_raw",
            shape=(),
            initializer=tf.keras.initializers.Constant(temp_raw_init),
            trainable=True,
        )
        self._last_gate_mean = self.add_weight(
            name="gate_mean_ema",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.target_mean),
            trainable=False,
        )
        self._last_gate_std = self.add_weight(
            name="gate_std_ema",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.2),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        temperature = tf.nn.softplus(self._temp_raw) + 1e-4
        gates = tf.sigmoid(inputs * temperature)
        gate_mean = tf.reduce_mean(gates)
        gate_std = tf.math.reduce_std(gates)
        self._last_gate_mean.assign(0.95 * self._last_gate_mean + 0.05 * gate_mean)
        self._last_gate_std.assign(0.95 * self._last_gate_std + 0.05 * gate_std)

        if self.balance_weight > 0:
            self.add_loss(self.balance_weight * tf.square(gate_mean - self.target_mean))
        if self.std_weight > 0 and self.min_std > 0:
            self.add_loss(self.std_weight * tf.square(tf.nn.relu(self.min_std - gate_std)))
        return gates

    def current_gate_mean(self):
        if self._last_gate_mean is None:
            return tf.constant(self.target_mean, dtype=tf.float32)
        return self._last_gate_mean

    def current_gate_std(self):
        if self._last_gate_std is None:
            return tf.constant(0.0, dtype=tf.float32)
        return self._last_gate_std


class ResidualBlend(layers.Layer):
    def __init__(
        self,
        alpha_init: float = 0.7,
        learnable: bool = True,
        adaptive_alpha: bool = True,
        alpha_ema_momentum: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        alpha_init = float(min(max(alpha_init, 1e-4), 1.0 - 1e-4))
        self.learnable = learnable
        self.adaptive_alpha = adaptive_alpha
        self.alpha_ema_momentum = float(min(max(alpha_ema_momentum, 0.0), 0.999))
        self._alpha_init = alpha_init
        self._alpha_logit = None
        self._alpha_const = tf.constant(alpha_init, dtype=tf.float32)
        self._alpha_ema = None

    def build(self, input_shape):
        if self.learnable:
            alpha_logit_init = tf.math.log(self._alpha_init / (1.0 - self._alpha_init))
            self._alpha_logit = self.add_weight(
                name="alpha_logit",
                shape=(),
                initializer=tf.keras.initializers.Constant(alpha_logit_init),
                trainable=True,
            )
        self._alpha_ema = self.add_weight(
            name="alpha_ema",
            shape=(),
            initializer=tf.keras.initializers.Constant(self._alpha_init),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            gated, residual, alpha_delta = inputs
        else:
            gated, residual = inputs
            alpha_delta = None

        if self.learnable and self._alpha_logit is not None:
            alpha_base = self._alpha_logit
            if self.adaptive_alpha and alpha_delta is not None:
                alpha = tf.sigmoid(alpha_base + alpha_delta)
            else:
                alpha = tf.sigmoid(alpha_base)
        else:
            alpha = self._alpha_const
        alpha = tf.clip_by_value(alpha, 1e-4, 1.0 - 1e-4)
        alpha_mean = tf.reduce_mean(alpha)
        self._alpha_ema.assign(self.alpha_ema_momentum * self._alpha_ema + (1.0 - self.alpha_ema_momentum) * alpha_mean)
        return alpha * gated + (1.0 - alpha) * residual

    def current_alpha(self):
        if self._alpha_ema is not None:
            return self._alpha_ema
        if self.learnable and self._alpha_logit is not None:
            return tf.sigmoid(self._alpha_logit)
        return self._alpha_const


def build_adaptive_refinement_model(
    pretrained_encoder: Model,
    num_classes: int = 7,
    alpha: float = 0.7,
    use_learnable_alpha: bool = True,
    dropout_rate: float = 0.4,
    use_dynamic_alpha: bool = True,
    alpha_ema_momentum: float = 0.9,
    gate_temperature_init: float = 1.5,
    gate_target_mean: float = 0.45,
    gate_balance_weight: float = 1e-3,
    gate_min_std: float = 0.08,
    gate_std_weight: float = 5e-4,
) -> Model:
    inputs = pretrained_encoder.input
    h = pretrained_encoder(inputs)

    d_path = layers.Dense(256, activation="relu", name="disc_dense")(h)
    d_path = layers.Dropout(float(dropout_rate), name="disc_dropout")(d_path)

    gate_logits = layers.Dense(256, activation=None, name="gate_dense")(h)
    g_path = SelectiveGate(
        temperature_init=gate_temperature_init,
        target_mean=gate_target_mean,
        balance_weight=gate_balance_weight,
        min_std=gate_min_std,
        std_weight=gate_std_weight,
        name="gate_values",
    )(gate_logits)

    gated = layers.Multiply(name="gate_mul")([d_path, g_path])
    alpha_inputs = [gated, h]
    if use_dynamic_alpha:
        alpha_delta = layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="alpha_delta",
        )(h)
        alpha_inputs.append(alpha_delta)
    refined = ResidualBlend(
        alpha_init=alpha,
        learnable=use_learnable_alpha,
        adaptive_alpha=use_dynamic_alpha,
        alpha_ema_momentum=alpha_ema_momentum,
        name="residual_blend",
    )(alpha_inputs)

    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(refined)
    return Model(inputs=inputs, outputs=outputs, name="adaptive_refinement_model")

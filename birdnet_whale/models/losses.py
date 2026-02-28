import tensorflow as tf


class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.07, name="sup_con_loss"):
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, projections, labels):
        projections = tf.math.l2_normalize(projections, axis=1)
        batch_size = tf.shape(projections)[0]

        similarity_matrix = tf.matmul(projections, projections, transpose_b=True) / self.temperature

        labels = tf.reshape(labels, (-1, 1))
        mask_same = tf.equal(labels, tf.transpose(labels))
        mask_not_self = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))
        mask_positive = tf.logical_and(mask_same, mask_not_self)

        exp_sim = tf.exp(similarity_matrix) * tf.cast(mask_not_self, tf.float32)
        denom = tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-8

        log_prob = similarity_matrix - tf.math.log(denom)
        num_pos = tf.reduce_sum(tf.cast(mask_positive, tf.float32), axis=1)
        loss_per = -tf.reduce_sum(tf.where(mask_positive, log_prob, 0.0), axis=1) / (num_pos + 1e-8)
        return tf.reduce_mean(loss_per)


class FocalLossWithLabelSmoothing(tf.keras.losses.Loss):
    def __init__(
        self,
        alpha=1.0,
        gamma=1.5,
        label_smoothing=0.1,
        num_classes=7,
        class_weights=None,
        name="focal_ls",
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = label_smoothing
        self.num_classes = num_classes
        if class_weights is None:
            self.class_weights = None
        else:
            weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
            weights = tf.reshape(weights, (-1,))
            if weights.shape[0] is not None and int(weights.shape[0]) != int(num_classes):
                raise ValueError(f"class_weights length mismatch: expected {num_classes}, got {int(weights.shape[0])}")
            self.class_weights = weights

    def call(self, y_true, y_pred):
        y_true = y_true * (1.0 - self.epsilon) + self.epsilon / self.num_classes
        ce = -y_true * tf.math.log(y_pred + 1e-7)
        weight = tf.pow(1.0 - y_pred, self.gamma)
        focal = self.alpha * weight * ce
        if self.class_weights is not None:
            focal = focal * self.class_weights[tf.newaxis, :]
        return tf.reduce_mean(tf.reduce_sum(focal, axis=-1))

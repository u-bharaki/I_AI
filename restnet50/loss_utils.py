import tensorflow as tf

from config import alpha

@tf.keras.utils.register_keras_serializable()
class CategoricalFocalLoss(tf.keras.losses.Loss):



    def __init__(self, gamma=2.0, alpha=alpha, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        loss = weight * ce
        return tf.reduce_sum(loss, axis=-1)
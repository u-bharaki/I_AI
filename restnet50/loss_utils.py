import tensorflow as tf
from config import alpha

@tf.keras.utils.register_keras_serializable()
class CategoricalFocalLoss(tf.keras.losses.Loss):

    # Fonksiyon adını __init__ (çift alt çizgi) olarak düzelttik
    def __init__(self, gamma=2.0, alpha=alpha, **kwargs):
        super().__init__(**kwargs) # Burada da çift alt çizgi olmalı
        self.gamma = gamma
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -y_true * tf.math.log(y_pred)
        # Artık self.alpha tanımlı olduğu için hata vermeyecektir
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        loss = weight * ce
        return tf.reduce_sum(loss, axis=-1)

    # Modelin kaydedilip tekrar yüklenmesinde sorun çıkmaması için get_config ekleyelim
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha.numpy().tolist()
        })
        return config
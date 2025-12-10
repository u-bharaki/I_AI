import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from config import *

def build_model(trainable=False):
    # EfficientNetB0 - ImageNet ağırlıklarıyla
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    )

    base_model.trainable = trainable

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # BatchNormalization: Öğrenmeyi hızlandırır
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x) # Overfitting önleyici

    x = Dense(256, activation="relu")(x)

    # İkinci katman koruması
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    out = Dense(NUM_CLASSES, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=out), base_model
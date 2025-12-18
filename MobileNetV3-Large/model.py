import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from config import *

def build_model(trainable=False):
    base_model = MobileNetV3Large(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = trainable

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Overfit önleyici ve kapasite artırıcı katmanlar
    x = BatchNormalization()(x)
    x = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) # L2 Regularization eklendi
    x = Activation('relu')(x)
    x = Dropout(0.5)(x) # %50 dropout ezberlemeyi engeller

    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)

    out = Dense(NUM_CLASSES, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=out), base_model
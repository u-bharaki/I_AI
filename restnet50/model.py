import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from config import *


def build_resnet50_model(trainable=True):
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    )

    base_model.trainable = trainable

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation="relu")(x)

    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    out = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=out)
    return model
# model.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from config import IMAGE_SIZE, NUM_CLASSES, CHANNELS


def build_model():
    """
    EfficientNetB0 kullanarak transfer öğrenimi modelini oluşturur ve döndürür.
    """
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)

    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model
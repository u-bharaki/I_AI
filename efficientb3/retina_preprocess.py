# retina_preprocess.py

import cv2
import numpy as np
import tensorflow as tf

def retina_preprocess(image):
    image = tf.cast(image * 255.0, tf.uint8)
    image_np = image.numpy()

    # Green channel extraction (retina literatürü)
    green = image_np[:, :, 1]

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    green_clahe = clahe.apply(green)

    # Merge back to 3-channel
    final = cv2.merge([green_clahe, green_clahe, green_clahe])

    final = tf.convert_to_tensor(final, dtype=tf.float32) / 255.0
    return final

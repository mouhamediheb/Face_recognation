

import tensorflow as tf
from tensorflow.keras.layers import Layer


#Custom L1 Distance Layer from Jupyter
class L1Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L1Dist, self).__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
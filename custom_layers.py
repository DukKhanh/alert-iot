import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers
import numpy as np
import math

# ===== REGISTER để load model không lỗi =====
from keras.saving import register_keras_serializable


# ==========================================
# SequenceToBatch
# ==========================================
@register_keras_serializable()
class SequenceToBatch(layers.Layer):
    def call(self, inputs):
        shape = tf.shape(inputs)
        return tf.reshape(inputs, [-1, shape[2], 1])

    def compute_output_shape(self, input_shape):
        return (None, input_shape[2], 1)


# ==========================================
# BatchToSequence
# ==========================================
@register_keras_serializable()
class BatchToSequence(layers.Layer):
    def __init__(self, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = seq_length

    def call(self, inputs):
        shape = tf.shape(inputs)
        return tf.reshape(inputs, [-1, self.seq_length, shape[-1]])

    def compute_output_shape(self, input_shape):
        return (None, self.seq_length, input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"seq_length": self.seq_length})
        return config


# ==========================================
# SincConv1D
# ==========================================
@register_keras_serializable()
class SincConv1D(layers.Layer):
    def __init__(self, num_filters=80, kernel_size=251, sample_rate=32000, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.l2_reg = l2_reg

    def build(self, input_shape):
        low_freq_mel = 80
        high_freq_mel = 2595 * math.log10(1 + (self.sample_rate / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.num_filters + 1)
        hz_points = 700 * (10**(mel_points / 2595) - 1)

        f1_init = (hz_points[:-1] / self.sample_rate).astype(np.float32)
        b1_init = ((hz_points[1:] - hz_points[:-1]) / self.sample_rate).astype(np.float32)

        self.f1 = self.add_weight(
            name='f1',
            shape=(self.num_filters,),
            initializer=initializers.Constant(f1_init),
            trainable=True
        )

        self.band = self.add_weight(
            name='band',
            shape=(self.num_filters,),
            initializer=initializers.Constant(b1_init),
            trainable=True,
            regularizer=regularizers.L2(self.l2_reg)
        )

        n_arr = np.arange(1, (self.kernel_size - 1) / 2 + 1, dtype=np.float32)
        self.n = tf.constant(n_arr)
        self.window = tf.constant(np.hamming(self.kernel_size).astype(np.float32))

        super().build(input_shape)

    def call(self, inputs):
        f1 = tf.abs(self.f1)
        band = tf.abs(self.band)
        f2 = f1 + band

        f1 = tf.reshape(f1, (self.num_filters, 1))
        f2 = tf.reshape(f2, (self.num_filters, 1))

        filter_right = (tf.sin(2 * math.pi * f2 * self.n) -
                        tf.sin(2 * math.pi * f1 * self.n)) / (math.pi * self.n)

        filter_center = 2 * (f2 - f1)
        filter_left = tf.reverse(filter_right, axis=[1])

        filters = tf.concat([filter_left, filter_center, filter_right], axis=1)
        filters = filters * self.window

        filters = tf.transpose(filters)
        filters = tf.reshape(filters, (self.kernel_size, 1, self.num_filters))

        return tf.nn.conv1d(inputs, filters, stride=1, padding='VALID')

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "sample_rate": self.sample_rate,
            "l2_reg": self.l2_reg
        })
        return config
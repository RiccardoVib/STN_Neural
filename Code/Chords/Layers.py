import tensorflow as tf
import math
from SSM import ResidualBlock
import numpy as np


class FiLM(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, **kwargs):
        super(FiLM, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.dense = tf.keras.layers.Dense(self.in_size * 2, use_bias=bias)
        self.glu = GLU(in_size=self.in_size)

    def call(self, x, c):
        c = self.dense(c)
        a, b = tf.split(c, 2, axis=self.dim)
        x = tf.multiply(a, x)
        x = tf.add(x, b)
        x = self.glu(x)
        return x

class TemporalFiLM(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, **kwargs):
        super(TemporalFiLM, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.gru = tf.keras.layers.GRU(self.in_size * 2, stateful=True, use_bias=bias)
        self.glu = GLU(in_size=self.in_size)

    def call(self, x, c):
        c = tf.expand_dims(c, axis=1)
        c = self.gru(c)
        a, b = tf.split(c, 2, axis=self.dim)
        x = tf.multiply(a, x)
        x = tf.add(x, b)
        x = self.glu(x)
        return x


class GLU(tf.keras.layers.Layer):
    def __init__(self, in_size, bias=True, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.in_size = in_size
        self.dense = tf.keras.layers.Dense(self.in_size * 2, use_bias=bias)

    def call(self, x):
        x = self.dense(x)
        out, gate = tf.split(x, 2, axis=self.dim)
        gate = tf.keras.activations.softsign(gate)
        x = tf.multiply(out, gate)
        return x

class EnhancementLayer(tf.keras.layers.Layer):
    def __init__(self, b_size, steps, bias=True, dim=-1, trainable=True, type=tf.float32):
        super(EnhancementLayer, self).__init__()
        self.bias = bias
        self.steps = steps
        self.dim = dim
        self.trainable = trainable
        self.type = type
        self.b_size = b_size

        act='tanh'
        self.proj = tf.keras.layers.Dense(32, batch_input_shape=(self.b_size, self.steps))
        self.conv = tf.keras.layers.Conv1D(16, 2, activation=act)
        self.conv2 = tf.keras.layers.Conv1D(8, 2, activation=act)
        self.conv3 = tf.keras.layers.Conv1D(4, 2, activation=act)
        self.conv3 = tf.keras.layers.Conv1D(1, 2, activation=act)

    def call(self, x):
        x = self.proj(x)
        x = tf.expand_dims(x, axis=-1)
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.squeeze(x, axis=-1)
        return x

import tensorflow as tf
from Layers import TemporalFiLM


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



class HarmonicEnhancementModel(tf.keras.Model):
    def __init__(self, b_size, steps, trainable=True, type=tf.float32):
        super(HarmonicEnhancementModel, self).__init__()

        self.b_size = b_size
        self.steps = steps
        self.trainable = trainable
        self.type = type
        
        self.enh = EnhancementLayer(self.b_size, self.steps, trainable=self.trainable, type=self.type)

        self.condproj = tf.keras.layers.Dense(32, batch_input_shape=(self.b_size, self.steps))
        self.film = TemporalFiLM(self.steps)
        
        self.outLay = tf.keras.layers.Dense(1)

    def __call__(self, inputs, training=False):

        freq_inputs = tf.reshape(inputs[0], [self.b_size, 3])  # BxDx1  #[self.B, 1]
        vel_inputs = tf.reshape(inputs[1], [self.b_size, 1])  # BxDx1  #[self.B, 1]
        k_inputs = tf.reshape(inputs[2], [self.b_size, 1])  # BxDx1  #[self.B, 1]
        S_inputs = tf.reshape(inputs[3], [self.b_size, self.steps])  # BxDx1  #[self.B, 1]
        
        inpu = tf.concat([freq_inputs, vel_inputs, k_inputs], axis=-1)
        cond = self.condproj(inpu)
        
        out = self.enh(S_inputs)
        out = self.film(out, cond)
        out = self.outLay(out)

        rms = tf.abs(tf.reduce_mean(tf.square(out), axis=-1, keepdims=True))

        return [out, rms]

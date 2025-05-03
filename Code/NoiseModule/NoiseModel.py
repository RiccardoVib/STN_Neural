import tensorflow as tf
from LayersNoise import NoiseGenerator
#import matplotlib.pyplot as plt

class NoiseModel(tf.keras.Model):
    def __init__(self, batch_size, num_steps=240, ir_size=2*(240-1), train_n=True, max_steps=2799.0, type=tf.float32):
        """
        Harmonic Enhancement Model
        :param batch_size: batch size
        :param num_steps: input size
        :param ir_size: size of impulse response to compute
        :param train_n: if train the layers
        """
        super(NoiseModel, self).__init__()
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.index_inputs = tf.keras.Input(batch_shape=(batch_size, 1), name='k_inputs')
        self.vel_inputs = tf.keras.Input(batch_shape=(batch_size, 1), name='vel_inputs')
        self.NoiseGenerator = NoiseGenerator(window_size=num_steps, ir_size=2*(num_steps-1), trainable=train_n, max_steps=max_steps, type=type)

    def __call__(self, inputs, training=False):
        index_inputs = tf.reshape(inputs[0], [self.batch_size, 1])
        vel_inputs = tf.reshape(inputs[1], [self.batch_size, 1])

        noise, mean = self.NoiseGenerator(vel_inputs, index_inputs)

        rms = tf.abs(tf.reduce_mean(tf.square(noise), axis=-1))
        return [rms, noise, mean]

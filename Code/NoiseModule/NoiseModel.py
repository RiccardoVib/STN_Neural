import tensorflow as tf
from Layers import NoiseGenerator

class NoiseModel(tf.keras.Model):
    def __init__(self, D=1, num_steps=240, train_n=True, type=tf.float32):
        super(NoiseModel, self).__init__()
        self.D = D
        self.freq_inputs = tf.keras.Input(batch_shape=(1, D), name='freq_input')
        self.index_inputs = tf.keras.Input(batch_shape=(1, D), name='k_inputs')
        self.vel_inputs = tf.keras.Input(batch_shape=(1, D), name='vel_inputs')
        self.NoiseGenerator = NoiseGenerator(window_size=num_steps, ir_size=num_steps*2, trainable=train_n, type=type)

    def __call__(self, inputs, training=False):
        freq_inputs = tf.reshape(inputs[0], [1, self.D, 1])
        index_inputs = tf.reshape(inputs[1], [1, self.D, 1])
        vel_inputs = tf.reshape(inputs[2], [1, self.D, 1])

        noise, mean = self.NoiseGenerator(freq_inputs, vel_inputs, index_inputs)

        noise = tf.math.reduce_sum(noise, axis=1, name='reduce_sum1')# sum notes

        rms = tf.abs(tf.reduce_mean(tf.square(noise), axis=-1))
        return [rms, noise, mean]
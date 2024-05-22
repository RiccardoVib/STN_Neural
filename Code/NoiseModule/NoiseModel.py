import tensorflow as tf
from Layers import NoiseGenerator

class NoiseModel(tf.keras.Model):
    def __init__(self, B, num_steps=240, ir_size=120, train_n=True, max_steps=2799.0, type=tf.float32):
        super(NoiseModel, self).__init__()
        self.B = B
        self.max_steps = max_steps
        self.freq_inputs = tf.keras.Input(batch_shape=(B, 1), name='freq_input')
        self.index_inputs = tf.keras.Input(batch_shape=(B, 1), name='k_inputs')
        self.vel_inputs = tf.keras.Input(batch_shape=(B, 1), name='vel_inputs')
        self.NoiseGenerator = NoiseGenerator(window_size=num_steps, ir_size=num_steps, trainable=train_n, max_steps=max_steps, type=type)

    def __call__(self, inputs, training=False):
        freq_inputs = tf.reshape(inputs[0], [self.B, 1])
        index_inputs = tf.reshape(inputs[1], [self.B, 1])
        vel_inputs = tf.reshape(inputs[2], [self.B, 1])

        noise, mean = self.NoiseGenerator(freq_inputs, vel_inputs, index_inputs)

        #noise = tf.math.reduce_sum(noise, axis=1, name='reduce_sum1')# sum notes

        rms = tf.abs(tf.reduce_mean(tf.square(noise), axis=-1))
        return [rms, noise, mean]

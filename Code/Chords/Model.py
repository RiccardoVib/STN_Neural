import tensorflow as tf
from Layers import TemporalFiLM


class EnhancementLayer(tf.keras.layers.Layer):
    def __init__(self, b_size, steps, bias=True, trainable=True, type=tf.float32):
        """
        Enhancement Layer
        :param b_size: batch size
        :param steps: input size
        :param bias: if use bias
        :param trainable: if train the layers
        """
        super(EnhancementLayer, self).__init__()
        self.bias = bias
        self.steps = steps
        self.trainable = trainable
        self.type = type
        self.b_size = b_size

        self.proj = tf.keras.layers.Dense(32, batch_input_shape=(self.b_size, self.steps))
        self.conv = tf.keras.layers.Conv1D(16, 2, activation='tanh')
        self.conv2 = tf.keras.layers.Conv1D(8, 2, activation='tanh')
        self.conv3 = tf.keras.layers.Conv1D(4, 2, activation='tanh')
        self.conv3 = tf.keras.layers.Conv1D(1, 2, activation='tanh')

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
        """
        Harmonic Enhancement Model
        :param b_size: batch size
        :param steps: input size
        :param trainable: if train the layers
        """

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

    def create_model_LSTM_DK1(units, mini_batch_size=2048, input_dim=1, b_size=2400, stateful=False):
        # Defining inputs
        inputs = tf.keras.layers.Input(
            batch_shape=(b_size, mini_batch_size, 1), name='input')
        freq_inputs = tf.keras.layers.Input(
            batch_shape=(b_size, mini_batch_size, 3), name='f')
        k_inputs = tf.keras.layers.Input(
            batch_shape=(b_size, mini_batch_size, 1), name='k')
        vel_inputs = tf.keras.layers.Input(
            batch_shape=(b_size, mini_batch_size, 1), name='v')

        inpu = tf.concat([freq_inputs, vel_inputs, k_inputs], axis=-1)
        cond = tf.keras.layers.Dense(32)(inpu)

        outputs = tf.keras.layers.LSTM(
            units, stateful=stateful, return_sequences=True, name="LSTM")(inputs)

        outputs = tf.keras.layers.LSTM(
            units, stateful=stateful, return_sequences=True, name="LSTM1")(outputs)

        outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
        outputs = outputs + inputs

        out = TemporalFiLM(1, stateful=stateful)(outputs, cond)

        # rms = tf.abs(tf.reduce_mean(tf.square(out), axis=1, keepdims=True))
        model = tf.keras.models.Model([freq_inputs, vel_inputs, k_inputs, inputs], out)

        model.summary()

        return model

def create_model_LSTM_DK1(units, mini_batch_size=2048, input_dim=1, b_size=2400, stateful=False):

    # Defining inputs
    inputs = tf.keras.layers.Input(
        batch_shape=(b_size, mini_batch_size, input_dim), name='input')
    freq_inputs = tf.keras.layers.Input(
        batch_shape=(b_size, mini_batch_size, 3), name='f')
    k_inputs = tf.keras.layers.Input(
        batch_shape=(b_size, mini_batch_size, 1), name='k')
    vel_inputs = tf.keras.layers.Input(
        batch_shape=(b_size, mini_batch_size, 1), name='v')

    input = tf.concat([freq_inputs, vel_inputs, k_inputs], axis=-1)
    cond = tf.keras.layers.Dense(32)(input)

    outputs = tf.keras.layers.LSTM(
                units, stateful=stateful, return_sequences=True, name="LSTM")(inputs)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
    outputs = outputs + inputs

    out = TemporalFiLM(1, stateful=stateful)(outputs, cond)

    #rms = tf.abs(tf.reduce_mean(tf.square(out), axis=1, keepdims=True))
    model = tf.keras.models.Model([freq_inputs, vel_inputs, k_inputs, inputs], out)

    model.summary()

    return model

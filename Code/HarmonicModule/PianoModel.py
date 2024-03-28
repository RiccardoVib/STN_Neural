import tensorflow as tf
from HarmonicLayers import InharmonicLayer, DecayLayer, SinuosoidsGenerator, DeltaLayer


class PianoModel(tf.keras.Model):
    def __init__(self, B=1, num_steps=240, harmonics=24, fs=24000, max_steps=2799.0, train_b=True, train_amps=True, train_rev=True, type=tf.float32):
        super(PianoModel, self).__init__()

        self.B = B
        self.harmonics = harmonics
        self.max_steps = max_steps
        self.train_rev = train_rev

        self.InharmonicModel = InharmonicLayer(harmonics=harmonics, B=self.B, Fs=fs, trainable=train_b, type=type)
        self.SinuosoidsGenerator = SinuosoidsGenerator(B=self.B, harmonics=harmonics, num_steps=num_steps, type=type)
        self.DecayModel = DecayLayer(B=self.B, harmonics=harmonics, trainable=train_amps, type=type)

        self.deltas = DeltaLayer(trainable=train_amps, type=type)

    def __call__(self, inputs, training=False):
        freq_inputs = tf.reshape(inputs[0], [self.B, 1]) #BxDx1  #[self.B, 1]
        index_inputs = tf.reshape(inputs[1], [self.B, 1]) #BxDx1  #[self.B, 1]
        vel_inputs = tf.reshape(inputs[2], [self.B, 1]) #BxDx1  #[self.B, 1]
        vel_inputs = tf.divide(vel_inputs, 120)

        ##### first polarization
        partials_predicted, f_n, B, final_n_c = self.InharmonicModel(freq_inputs)
        partials_xt, f, t = self.SinuosoidsGenerator(partials_predicted, index_inputs)

        alfa_v, alfa_h, attackTime_s = self.DecayModel(f_n, vel_inputs, f, t, index_inputs / self.max_steps)

        all_harmonics = tf.sin(partials_xt)
        all_harmonics = tf.multiply(alfa_v, all_harmonics)

        all_harmonics = tf.math.reduce_sum(all_harmonics, axis=-1, name='reduce_harmonics')  # sum harmonics

     
        ##### second polarization +/- 0.1/0.2 hz
        delta_T2 = self.deltas(f_n, vel_inputs)
        partials_h = tf.add(delta_T2, freq_inputs)
        partials_h = tf.keras.layers.Multiply()([final_n_c, partials_h])  # BxDxH
        partials_xt_h, _, _ = self.SinuosoidsGenerator(partials_h, index_inputs)

        all_harmonics_h = tf.sin(partials_xt_h)
        all_harmonics_h = tf.multiply(alfa_h, all_harmonics_h)
        all_harmonics_h = tf.math.reduce_sum(all_harmonics_h, axis=-1, name='reduce_harmonics')  # sum harmonics
        all_harmonics = tf.add(all_harmonics_h, all_harmonics)

        alfa = tf.math.reduce_max(tf.abs(all_harmonics), axis=-1, keepdims=True)
        rms = tf.abs(tf.reduce_mean(tf.square(all_harmonics), axis=-1, keepdims=True))

        return [partials_predicted[:, :6], B, all_harmonics, rms, alfa, attackTime_s]
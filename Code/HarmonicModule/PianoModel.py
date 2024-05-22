import tensorflow as tf
from HarmonicLayers import InharmonicLayer, DecayLayer, SinuosoidsGenerator, DeltaLayer

class PianoModel(tf.keras.Model):
    def __init__(self, B=1, num_steps=240, harmonics=24, fs=24000, max_steps=2799.0, train_b=True, train_amps=True,
                 train_rev=False, train_=False, mul=1, type=tf.float32):
        super(PianoModel, self).__init__()

        self.B = B
        self.harmonics = harmonics
        self.max_steps = max_steps
        self.train_rev = train_rev
        self.train_ = train_
        self.mul=mul
        self.h_even = self.harmonics# - 10
        self.h_odd = self.h_even*2 - 2

        self.InharmonicModel = InharmonicLayer(harmonics=harmonics, B=self.B, Fs=fs, trainable=train_b, type=type)
        self.SinuosoidsGenerator = SinuosoidsGenerator(B=self.B, harmonics=harmonics, num_steps=num_steps, type=type)
        self.SinuosoidsGenerator_odd = SinuosoidsGenerator(B=self.B, harmonics=self.h_odd, num_steps=num_steps, type=type)
        self.SinuosoidsGenerator_even = SinuosoidsGenerator(B=self.B, harmonics=self.h_even, num_steps=num_steps, type=type)
        self.DecayModel = DecayLayer(B=self.B, harmonics=harmonics, trainable=train_amps, mul=self.mul, type=type)
        self.deltas = DeltaLayer(trainable=train_amps, mul=self.mul, type=type)
    
    def __call__(self, inputs, training=False):
        freq_inputs = tf.reshape(inputs[0], [self.B, 1])  # BxDx1  #[self.B, 1]
        index_inputs = tf.reshape(inputs[1], [self.B, 1])  # BxDx1  #[self.B, 1]
        vel_inputs = tf.reshape(inputs[2], [self.B, 1])  # BxDx1  #[self.B, 1]
        vel_inputs = tf.divide(vel_inputs, 120)
        g=0#9
        
        ######Compute Frequencies

        ##### first polarization
        partials_predicted, f_n, B, final_n_c = self.InharmonicModel(freq_inputs)
        amplitude_envelopes = tf.where(tf.greater_equal(partials_predicted, 11000),
                                       tf.zeros_like(partials_predicted), 1)
        partials_predicted_n = amplitude_envelopes * partials_predicted

        #### Odd
        partials_predicted_odd = []
        for i in range(g, partials_predicted.shape[-1]-1):
                partials_predicted_odd.append(tf.add(partials_predicted[:, i], partials_predicted[:, i+1]))
                partials_predicted_odd.append(tf.abs(tf.add(partials_predicted[:, i], -partials_predicted[:, i+1])))
        partials_predicted_odd = tf.transpose(tf.convert_to_tensor(partials_predicted_odd))
        amplitude_envelopes = tf.where(tf.greater_equal(partials_predicted_odd, 11000), tf.zeros_like(partials_predicted_odd), 1)
        partials_predicted_odd_n = amplitude_envelopes*partials_predicted_odd

        #### even
        partials_predicted_even = tf.multiply(partials_predicted[:, g:], 2.)
        amplitude_envelopes = tf.where(tf.greater_equal(partials_predicted_even, 11000), tf.zeros_like(partials_predicted_even), 1)
        partials_predicted_even_n = amplitude_envelopes*partials_predicted_even

        ##### second polarization +/- 0.1/0.2 hz
        delta_T2 = self.deltas(f_n, vel_inputs)
        partials_h = tf.add(delta_T2, freq_inputs)
        partials_h = tf.keras.layers.Multiply()([final_n_c, partials_h])  # BxDxH
        amplitude_envelopes = tf.where(tf.greater_equal(partials_h, 11000), tf.zeros_like(partials_h), 1)
        partials_h_n = amplitude_envelopes*partials_h

        #### Odd
        partials_predicted_odd2 = []
        for i in range(g, partials_h.shape[-1]-1):
            partials_predicted_odd2.append(tf.add(partials_h[:, i], partials_h[:, i + 1]))
            partials_predicted_odd2.append(tf.abs(tf.add(partials_h[:, i], -partials_h[:, i + 1])))
        partials_predicted_odd2 = tf.transpose(tf.convert_to_tensor(partials_predicted_odd2))
        amplitude_envelopes = tf.where(tf.greater_equal(partials_predicted_odd2, 11000), tf.zeros_like(partials_predicted_odd2), 1)
        partials_predicted_odd2_n = amplitude_envelopes*partials_predicted_odd2

        #### even2
        partials_predicted_even2 = tf.multiply(partials_h[:, g:], 2.)
        amplitude_envelopes = tf.where(tf.greater_equal(partials_predicted_even2, 11000), tf.zeros_like(partials_predicted_even2), 1)
        partials_predicted_even2_n = amplitude_envelopes*partials_predicted_even2

        #### Odd mixed
        partials_predicted_odd3 = []
        for i in range(g, partials_h.shape[-1]-1):
            partials_predicted_odd3.append(tf.add(partials_predicted[:, i], partials_h[:, i + 1]))
            partials_predicted_odd3.append(tf.abs(tf.add(partials_predicted[:, i], -partials_h[:, i + 1])))
        partials_predicted_odd3 = tf.transpose(tf.convert_to_tensor(partials_predicted_odd3))
        #### even
        amplitude_envelopes = tf.where(tf.greater_equal(partials_predicted_odd3, 11000), tf.zeros_like(partials_predicted_odd3), 1)
        partials_predicted_odd3_n = amplitude_envelopes*partials_predicted_odd3

        #####Compute Sines

        partials_xt, f, t = self.SinuosoidsGenerator(partials_predicted_n, index_inputs)
        partials_xt_h, f_h, _ = self.SinuosoidsGenerator(partials_h_n, index_inputs)
        partials_xt_even, f_even, t_even = self.SinuosoidsGenerator_even(partials_predicted_even_n, index_inputs)
        partials_xt_even2, f_even2, _ = self.SinuosoidsGenerator_even(partials_predicted_even2_n, index_inputs)
        partials_xt_odd, f_odd, t_odd = self.SinuosoidsGenerator_odd(partials_predicted_odd_n, index_inputs)
        partials_xt_odd2, f_odd2, _ = self.SinuosoidsGenerator_odd(partials_predicted_odd2_n, index_inputs)
        partials_xt_odd3, f_odd3, _ = self.SinuosoidsGenerator_odd(partials_predicted_odd3_n, index_inputs)

        partials = [f_even, f_even2, f_odd, f_odd2, f_odd3]
        
        alfa_v, alfa_h, attackTime_s, attackTime_t, decays_long = self.DecayModel(f_n, vel_inputs, [f, f_h], t, index_inputs / self.max_steps, partials, t_even, t_odd)
        all_harmonics = tf.sin(partials_xt)
        all_harmonics = tf.multiply(alfa_v, all_harmonics)
        all_harmonics = tf.math.reduce_sum(all_harmonics, axis=-1, name='reduce_harmonics')  # sum harmonics


        all_harmonics_h = tf.sin(partials_xt_h)
        all_harmonics_h = tf.multiply(alfa_h, all_harmonics_h)
        all_harmonics_h = tf.math.reduce_sum(all_harmonics_h, axis=-1, name='reduce_harmonics')  # sum harmonics
        all_harmonics = tf.add(all_harmonics_h, all_harmonics)

        ## long
        #all_harmonics_even = tf.sin(partials_xt_even)
        #all_harmonics_even = tf.multiply(decays_long[0], all_harmonics_even)
        #all_harmonics_even = tf.math.reduce_sum(all_harmonics_even, axis=-1, name='reduce_harmonics')  # sum harmonics
        #all_harmonics = tf.add(all_harmonics_even, all_harmonics)

        #all_harmonics_even = tf.sin(partials_xt_even2)
        #all_harmonics_even = tf.multiply(decays_long[1], all_harmonics_even)
        #all_harmonics_even = tf.math.reduce_sum(all_harmonics_even, axis=-1, name='reduce_harmonics')  # sum harmonics
        #all_harmonics = tf.add(all_harmonics_even, all_harmonics)

        all_harmonics_odd = tf.sin(partials_xt_odd)
        all_harmonics_odd = tf.multiply(decays_long[2], all_harmonics_odd)
        all_harmonics_odd = tf.math.reduce_sum(all_harmonics_odd, axis=-1, name='reduce_harmonics')  # sum harmonics
        all_harmonics = tf.add(all_harmonics_odd, all_harmonics)

        all_harmonics_odd = tf.sin(partials_xt_odd2)
        all_harmonics_odd = tf.multiply(decays_long[3], all_harmonics_odd)
        all_harmonics_odd = tf.math.reduce_sum(all_harmonics_odd, axis=-1, name='reduce_harmonics')  # sum harmonics
        all_harmonics = tf.add(all_harmonics_odd, all_harmonics)

        all_harmonics_odd = tf.sin(partials_xt_odd3)
        all_harmonics_odd = tf.multiply(decays_long[4], all_harmonics_odd)
        all_harmonics_odd = tf.math.reduce_sum(all_harmonics_odd, axis=-1, name='reduce_harmonics')  # sum harmonics
        all_harmonics = tf.add(all_harmonics_odd, all_harmonics)

        all_harmonics = tf.multiply(attackTime_t, all_harmonics)
        if self.train_:
            all_harmonics = self.enh(all_harmonics)


        alfa = tf.math.reduce_max(tf.abs(all_harmonics), axis=-1, keepdims=True)
        rms = tf.abs(tf.reduce_mean(tf.square(all_harmonics), axis=-1, keepdims=True))

        return [partials_predicted[:, :6], B, all_harmonics, rms, alfa, attackTime_s]

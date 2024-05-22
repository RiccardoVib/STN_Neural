import tensorflow as tf
from einops import rearrange, repeat
import numpy as np
import math

def selective_scan(u, delta, A, B, C, D):
    # first step of A_bar = exp(ΔA), i.e., ΔA
    dA = tf.einsum('bld,dn->bldn', delta, A)
    dB_u = tf.einsum('bld,bld,bln->bldn', delta, u, B)

    dA_cumsum = tf.pad(
        dA[:, 1:], [[0, 0], [1, 1], [0, 0], [0, 0]])[:, 1:, :, :]

    dA_cumsum = tf.reverse(dA_cumsum, axis=[1])  # Flip along axis 1

    # Cumulative sum along all the input tokens, parallel prefix sum,
    # calculates dA for all the input tokens parallely
    dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)

    # second step of A_bar = exp(ΔA), i.e., exp(ΔA)
    dA_cumsum = tf.exp(dA_cumsum)
    dA_cumsum = tf.reverse(dA_cumsum, axis=[1])  # Flip back along axis 1

    x = dB_u * dA_cumsum
    # 1e-12 to avoid division by 0
    x = tf.math.cumsum(x, axis=1 ) /(dA_cumsum + 1e-12)

    y = tf.einsum('bldn,bln->bld', x, C)

    return y + u * D


class MambaBlock(tf.keras.layers.Layer):
    def __init__(self, layer_id, model_input_dims, model_internal_dim, conv_use_bias, dense_use_bias, conv_kernel_size, delta_t_rank, model_states, **kwargs):
        super(MambaBlock, self).__init__(**kwargs)
        self.layer_id = layer_id
        self.model_internal_dim = model_internal_dim
        self.model_input_dims = model_input_dims
        self.conv_use_bias = conv_use_bias
        self.dense_use_bias = dense_use_bias
        self.conv_kernel_size = conv_kernel_size
        self.delta_t_rank = delta_t_rank
        self.model_states = model_states

        self.in_projection = tf.keras.layers.Dense(
            self.model_internal_dim * 2,
            input_shape=(self.model_input_dims,), use_bias=False)

        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.model_internal_dim,
            use_bias=self.conv_use_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.model_internal_dim,
            data_format='channels_first',
            padding='causal'
        )

        # this layer takes in current token 'x'
        # and outputs the input-specific Δ, B, C (according to S6)
        self.x_projection = tf.keras.layers.Dense(self.delta_t_rank + self.model_states * 2, use_bias=False)

        # this layer projects Δ from delta_t_rank to the mamba internal
        # dimension
        self.delta_t_projection = tf.keras.layers.Dense(self.model_internal_dim,
                                               input_shape=(self.delta_t_rank,), use_bias=True)

        self.A = repeat(
            tf.range(1, self.model_states + 1, dtype=tf.float32),
            'n -> d n', d=self.model_internal_dim)

        self.A_log = tf.Variable(
            tf.math.log(self.A),
            trainable=True, dtype=tf.float32,
            name=f"SSM_A_log_{self.layer_id}")

        self.D = tf.Variable(
            np.ones(self.model_internal_dim),
            trainable=True, dtype=tf.float32,
            name=f"SSM_D_{self.layer_id}")

        self.out_projection = tf.keras.layers.Dense(
            self.model_input_dims,
            input_shape=(self.model_internal_dim,),
            use_bias=self.dense_use_bias)

    def call(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba pape.
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        """

        (batch_size, seq_len, dimension) = x.shape

        x_and_res = self.in_projection(x)  # shape = (batch, seq_len, 2 * model_internal_dimension)
        (x, res) = tf.split(x_and_res,
                            [self.model_internal_dim,
                             self.model_internal_dim], axis=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = tf.nn.swish(x)
        #x = tf.nn.softsign(x)
        y = self.ssm(x)
        y = y * tf.nn.swish(res)
        #y = y * tf.nn.softsign(res)
        return self.out_projection(y)

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper
            - run_SSM(A, B, C, u) in The Annotated S4
            Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -tf.exp(tf.cast(self.A_log, tf.float32))  # shape -> (d_in, n)
        D = tf.cast(self.D, tf.float32)

        x_dbl = self.x_projection(x)  # shape -> (batch, seq_len, delta_t_rank + 2*n)

        (delta, B, C) = tf.split(
            x_dbl,
            num_or_size_splits=[self.delta_t_rank, n, n],
            axis=-1)  # delta.shape -> (batch, seq_len) & B, C shape -> (batch, seq_len, n)

        delta = tf.nn.softplus(self.delta_t_projection(delta))  # shape -> (batch, seq_len, model_input_dim)

        return selective_scan(x, delta, A, B, C, D)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, layer_id, model_input_dims, model_internal_dim, conv_use_bias, dense_use_bias, conv_kernel_size, delta_t_rank, model_states, **kwargs):
        super().__init__(**kwargs)
        self.mixer = MambaBlock(layer_id, model_input_dims, model_internal_dim, conv_use_bias, dense_use_bias, conv_kernel_size, delta_t_rank, model_states)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        """
        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        return self.mixer(self.norm(x)) + x
        #return self.mixer(x) + x


def init_model(opt, b_size, layer_id=-1, input_dims=64, model_input_dims=4, conv_use_bias=True, dense_use_bias=True, projection_expand_factor=2, conv_kernel_size=4, model_states=8, num_layers=1, dropout_rate=0., loss='mse', metrics=['mse']):
    """

    :type model_input_dims: object
    """
    inp = tf.keras.layers.Input(batch_shape=(b_size, input_dims), name='input_ids')
    x = tf.keras.layers.Dense(model_input_dims)(inp)
    x = tf.expand_dims(x, axis=1)

    model_internal_dim = int(projection_expand_factor * model_input_dims)
    delta_t_rank = math.ceil(model_input_dims / 2)#16
    if layer_id == -1:
        layer_id = np.round(np.random.randint(0, 1000), 4)

    for i in range(num_layers):
        x = ResidualBlock(layer_id, model_input_dims, model_internal_dim, conv_use_bias, dense_use_bias, conv_kernel_size, delta_t_rank, model_states, name=f"Residual_{i}")(x)
        #x = tf.keras.layers.Dropout(dropout_rate)(x)  # for regularization

    x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)  # normalization layer
    x = tf.keras.layers.Dense(model_states//2, activation=tf.nn.gelu)(x)
    x = tf.squeeze(x, axis=1)

    output_layer = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=[inp], outputs=output_layer, name='Bamba')
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    model.summary()
    return model



"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

utils.py (Modified for Static + Temporal Features)

(1) train_test_divide: Divide train and test data.
(2) extract_time: Returns Maximum sequence length and each sequence length.
(3) rnn_cell: Basic RNN Cell.
(4) random_generator: random vector generator
(5) batch_generator: mini-batch generator
"""

## Necessary Packages
import numpy as np
import tensorflow as tf


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.default_rng().permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.default_rng().permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    )


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data (temporal)

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


class LayerNormLSTMCell(tf.keras.layers.LSTMCell):
    """LSTMCell with Layer Normalization.
    (No changes from original)
    """

    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        layer_norm_epsilon=1e-5,
        **kwargs
    ):
        super().__init__(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            **kwargs
        )
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        super().build(input_shape)
        self.ln_i = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.ln_c = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.ln_o = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.ln_cell_state = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )

    def call(self, inputs, states):
        h_tm1, c_tm1 = states
        z = tf.keras.backend.dot(inputs, self.kernel)
        z += tf.keras.backend.dot(h_tm1, self.recurrent_kernel)
        z = tf.keras.backend.bias_add(z, self.bias)
        z_i, z_f, z_c, z_o = tf.split(z, 4, axis=1)
        z_i = self.ln_i(z_i)
        z_f = self.ln_f(z_f)
        z_c = self.ln_c(z_c)
        z_o = self.ln_o(z_o)
        i = self.recurrent_activation(z_i)
        f = self.recurrent_activation(z_f)
        c = self.activation(z_c)
        o = self.recurrent_activation(z_o)
        new_c = f * c_tm1 + i * c
        new_c_norm = self.ln_cell_state(new_c)
        new_h = o * self.activation(new_c_norm)
        return new_h, [new_h, new_c]

    def get_config(self):
        config = super().get_config()
        config.update({"layer_norm_epsilon": self.layer_norm_epsilon})
        return config


def rnn_cell(module_name, hidden_dim):
    """Basic RNN Cell.
    (No changes from original)

    Args:
      - module_name: gru, lstm, or lstmLN

    Returns:
      - rnn_cell: RNN Cell
    """
    assert module_name in ["gru", "lstm", "lstmLN"]

    if module_name == "gru":
        rnn_cell = tf.keras.layers.GRUCell(units=hidden_dim, activation="tanh")
    elif module_name == "lstm":
        rnn_cell = tf.keras.layers.LSTMCell(units=hidden_dim, activation="tanh")
    elif module_name == "lstmLN":
        rnn_cell = LayerNormLSTMCell(units=hidden_dim, activation="tanh")
    return rnn_cell


def random_generator(batch_size, z_dim_s, z_dim_x, T_mb, max_seq_len):
    """Random vector generation.

    Args:
      - batch_size: size of the random vector
      - z_dim_s: dimension of static random vector
      - z_dim_x: dimension of temporal random vector
      - T_mb: time information for the random vector
      - max_seq_len: maximum sequence length

    Returns:
      - Z_S_mb: generated static random vector
      - Z_T_mb: generated temporal random vector
    """
    # Static random vector
    Z_S_mb = np.random.default_rng().uniform(0.0, 1, [batch_size, z_dim_s])

    # Temporal random vector
    Z_T_mb = list()
    for i in range(batch_size):
        temp = np.zeros([max_seq_len, z_dim_x])
        temp_Z = np.random.default_rng().uniform(0.0, 1, [T_mb[i], z_dim_x])
        temp[: T_mb[i], :] = temp_Z
        Z_T_mb.append(temp)
    return Z_S_mb, Z_T_mb


def batch_generator(data_s, data_x, time, batch_size):
    """Mini-batch generator.

    Args:
      - data_s: static time-series data
      - data_x: temporal time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - S_mb: static data in each batch
      - X_mb: temporal data in each batch
      - T_mb: time information in each batch
    """
    no = len(data_x)
    idx = np.random.default_rng().permutation(no)
    train_idx = idx[:batch_size]

    S_mb = list(data_s[i] for i in train_idx)
    X_mb = list(data_x[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return S_mb, X_mb, T_mb


def batch_generator1(data, time, batch_size):
    """Mini-batch generator.

    Args:
      - data: time-series data
      - time: time information
      - batch_size: the number of samples in each batch

    Returns:
      - X_mb: time-series data in each batch
      - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.default_rng().permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb

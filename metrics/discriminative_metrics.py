"""
discriminative_metrics.py

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

# Necessary Packages
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator1


class Discriminator(Model):
    """
    Keras Model for the post-hoc discriminator.
    Replaces the TF 1.x static graph definition.
    """

    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.gru_cell = layers.GRUCell(units=hidden_dim, activation="tanh")
        # The RNN layer wraps the cell, matching tf.nn.dynamic_rnn behavior
        self.rnn = layers.RNN(self.gru_cell, return_sequences=False)
        # Replaces tf.contrib.layers.fully_connected
        self.dense = layers.Dense(1, activation=None)

    def call(self, x, t):
        """
        Args:
          - x: time-series data (Batch, MaxSeqLen, Dim)
          - t: time information (Batch,)
        """
        # Create a boolean mask from sequence lengths
        mask = tf.sequence_mask(t, maxlen=tf.shape(x)[1])

        # Pass the mask to the RNN layer
        # This replicates the behavior of 'sequence_length' in tf.nn.dynamic_rnn
        d_last_states = self.rnn(x, mask=mask)

        # Get logits
        y_hat_logit = self.dense(d_last_states)
        y_hat = tf.nn.sigmoid(y_hat_logit)

        return y_hat_logit, y_hat


def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data

    Args:
      - ori_data: original data (list of 2D numpy arrays)
      - generated_data: generated synthetic data (list of 2D numpy arrays)

    Returns:
      - discriminative_score: np.abs(classification accuracy - 0.5)
    """

    # Basic Parameters
    # Get dim from the first sample. Assumes all samples have same dimension.
    dim = ori_data[0].shape[1]

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    ## Builde a post-hoc RNN discriminator network
    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128

    # Instantiate the Keras model
    discriminator_model = Discriminator(hidden_dim)

    # Optimizer and Loss
    # Replaces tf.train.AdamOptimizer()
    d_optimizer = optimizers.Adam()
    # Replaces tf.nn.sigmoid_cross_entropy_with_logits
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    ## Train the discriminator
    # Train/test division for both original and generated data
    (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    ) = train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # Define the training step as a tf.function for performance
    @tf.function
    def train_step(X_mb, T_mb, X_hat_mb, T_hat_mb):
        with tf.GradientTape() as tape:
            # Get logits for real and fake data
            y_logit_real, _ = discriminator_model(X_mb, T_mb)
            y_logit_fake, _ = discriminator_model(X_hat_mb, T_hat_mb)

            # Calculate loss
            d_loss_real = bce_loss(tf.ones_like(y_logit_real), y_logit_real)
            d_loss_fake = bce_loss(tf.zeros_like(y_logit_fake), y_logit_fake)
            d_loss = d_loss_real + d_loss_fake

        # Get gradients and apply them
        gradients = tape.gradient(d_loss, discriminator_model.trainable_variables)
        d_optimizer.apply_gradients(
            zip(gradients, discriminator_model.trainable_variables)
        )
        return d_loss

    # Training loop
    for itt in range(iterations):
        # Batch setting
        X_mb_list, T_mb = batch_generator1(train_x, train_t, batch_size)
        X_hat_mb_list, T_hat_mb = batch_generator1(train_x_hat, train_t_hat, batch_size)

        # The `utils.batch_generator1` returns a list of arrays of variable length
        # We must pad them to a consistent 3D array shape.
        X_mb = np.zeros([batch_size, max_seq_len, dim])
        for i in range(batch_size):
            if T_mb[i] > 0:  # Handle potential empty sequences
                X_mb[i, : T_mb[i], :] = X_mb_list[i]

        X_hat_mb = np.zeros([batch_size, max_seq_len, dim])
        for i in range(batch_size):
            if T_hat_mb[i] > 0:
                X_hat_mb[i, : T_hat_mb[i], :] = X_hat_mb_list[i]

        # Convert T lists to numpy arrays
        T_mb = np.array(T_mb)
        T_hat_mb = np.array(T_hat_mb)

        # Train discriminator
        step_d_loss = train_step(X_mb, T_mb, X_hat_mb, T_hat_mb)

    ## Test the performance on the testing set

    test_x_padded = np.zeros([len(test_x), max_seq_len, dim])
    for i in range(len(test_x)):
        if test_t[i] > 0:
            test_x_padded[i, : test_t[i], :] = test_x[i]

    test_x_hat_padded = np.zeros([len(test_x_hat), max_seq_len, dim])
    for i in range(len(test_x_hat)):
        if test_t_hat[i] > 0:
            test_x_hat_padded[i, : test_t_hat[i], :] = test_x_hat[i]

    # Convert T lists to numpy arrays
    test_t_arr = np.array(test_t)
    test_t_hat_arr = np.array(test_t_hat)

    # Get predictions
    _, y_pred_real_curr = discriminator_model(test_x_padded, test_t_arr)
    _, y_pred_fake_curr = discriminator_model(test_x_hat_padded, test_t_hat_arr)

    # Convert from EagerTensors to numpy arrays
    y_pred_real_curr = y_pred_real_curr.numpy()
    y_pred_fake_curr = y_pred_fake_curr.numpy()

    y_pred_final = np.squeeze(
        np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis=0)
    )
    y_label_final = np.concatenate(
        (
            np.ones(
                [
                    len(y_pred_real_curr),
                ]
            ),
            np.zeros(
                [
                    len(y_pred_fake_curr),
                ]
            ),
        ),
        axis=0,
    )

    # Compute the accuracy
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score

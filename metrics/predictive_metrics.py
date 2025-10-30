"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

# Necessary Packages
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np
from sklearn.metrics import mean_absolute_error
from utils import extract_time


class Predictor(Model):
    """
    Keras Model for the post-hoc predictor.
    Replaces the TF 1.x static graph definition.
    """

    def __init__(self, hidden_dim):
        super(Predictor, self).__init__()
        self.gru_cell = layers.GRUCell(units=hidden_dim, activation="tanh")
        # return_sequences=True to predict at each time step
        self.rnn = layers.RNN(self.gru_cell, return_sequences=True)
        # TimeDistributed Dense layer to get 1-dim output at each step
        self.dense = layers.Dense(1, activation=None)

    def call(self, x, t):
        """
        Args:
          - x: time-series data (Batch, MaxSeqLen-1, Dim-1)
          - t: time information (Batch,)
        """
        # Create a boolean mask from sequence lengths
        mask = tf.sequence_mask(t, maxlen=tf.shape(x)[1])

        # Pass the mask to the RNN layer
        rnn_outputs = self.rnn(x, mask=mask)

        # Get logits
        y_hat_logit = self.dense(rnn_outputs)
        y_hat = tf.nn.sigmoid(y_hat_logit)  # As in original code

        return y_hat


def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
      - ori_data: original data (list of 2D numpy arrays)
      - generated_data: generated synthetic data (list of 2D numpy arrays)

    Returns:
      - predictive_score: MAE of the predictions on the original data
    """

    # Basic Parameters
    no = len(ori_data)
    if no == 0:
        return 0.0
    dim = ori_data[0].shape[1]

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    # [FIXED BUG]: Was using ori_data twice
    generated_time, generated_max_seq_len = extract_time(generated_data)
    # The predictive model inputs are 1 step shorter
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    ## Builde a post-hoc RNN predictive network
    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 5000  # As in original
    batch_size = 128

    # Instantiate the Keras model
    predictor_model = Predictor(hidden_dim)

    # Optimizer
    p_optimizer = optimizers.Adam()

    # Define the training step as a tf.function for performance
    @tf.function
    def train_step(X, Y, T):
        with tf.GradientTape() as tape:
            # Get predictions
            y_pred = predictor_model(X, T)

            # [FIXED BUG]: Create mask to compute loss only on valid data
            mask = tf.sequence_mask(T, maxlen=tf.shape(X)[1], dtype=tf.float32)
            # Add an extra dimension to mask to match Y's shape
            mask = tf.expand_dims(mask, -1)  # Shape: (batch, seq_len-1, 1)

            # Calculate loss: tf.losses.absolute_difference
            loss_unmasked = tf.abs(Y - y_pred)
            loss_masked = loss_unmasked * mask

            # Compute mean loss only over non-padded elements
            p_loss = tf.reduce_sum(loss_masked) / tf.reduce_sum(mask)

        # Get gradients and apply them
        gradients = tape.gradient(p_loss, predictor_model.trainable_variables)
        p_optimizer.apply_gradients(zip(gradients, predictor_model.trainable_variables))
        return p_loss

    ## Training
    # Training using Synthetic dataset
    for itt in range(iterations):

        # Set mini-batch
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]

        # [FIXED BUG]: Initialize padded arrays for the batch
        X_mb = np.zeros([batch_size, max_seq_len - 1, dim - 1], dtype=np.float32)
        Y_mb = np.zeros([batch_size, max_seq_len - 1, 1], dtype=np.float32)
        T_mb = np.zeros([batch_size], dtype=np.int32)

        for i, mb_idx in enumerate(train_idx):
            # T_i is the length of the *input* sequence (original_len - 1)
            T_i = generated_time[mb_idx] - 1
            if T_i <= 0:
                continue  # Skip empty or single-step sequences

            X_mb[i, :T_i, :] = generated_data[mb_idx][:-1, : (dim - 1)]
            Y_mb[i, :T_i, :] = np.reshape(
                generated_data[mb_idx][1:, (dim - 1)], [T_i, 1]
            )
            T_mb[i] = T_i

        # Train predictor
        step_p_loss = train_step(X_mb, Y_mb, T_mb)

    ## Test the trained model on the original data

    # [FIXED BUG]: Pad the entire test set for prediction
    X_test = np.zeros([no, max_seq_len - 1, dim - 1], dtype=np.float32)
    Y_test = np.zeros([no, max_seq_len - 1, 1], dtype=np.float32)
    T_test = np.zeros([no], dtype=np.int32)

    for i in range(no):
        T_i = ori_time[i] - 1
        if T_i <= 0:
            continue

        X_test[i, :T_i, :] = ori_data[i][:-1, : (dim - 1)]
        Y_test[i, :T_i, :] = np.reshape(ori_data[i][1:, (dim - 1)], [T_i, 1])
        T_test[i] = T_i

    # Prediction
    pred_Y_curr = predictor_model(X_test, T_test).numpy()

    # Compute the performance in terms of MAE
    # [FIXED BUG]: Calculate MAE only on valid (non-padded) data
    MAE_temp = 0
    for i in range(no):
        T_i = T_test[i]
        if T_i <= 0:
            continue

        # Get the non-padded slices for true and pred
        Y_true_i = Y_test[i, :T_i, :]
        Y_pred_i = pred_Y_curr[i, :T_i, :]

        MAE_temp += mean_absolute_error(Y_true_i, Y_pred_i)

    predictive_score = MAE_temp / no

    return predictive_score

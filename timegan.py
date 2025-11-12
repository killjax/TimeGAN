import tensorflow as tf
import os
from tensorflow.keras.layers import (
    Input,
    Dense,
    RNN,
    Bidirectional,
    Concatenate,
    RepeatVector,
    Lambda,
)
from tensorflow.keras.models import Model
import numpy as np

from utils import (
    extract_time,
    rnn_cell,
    random_generator,
    batch_generator,
    LayerNormLSTMCell,
)


# Helper function
def MinMaxScaler(data):
    """Min-Max Normalizer.
    Args:
      - data: raw data (can be 2D for static, 3D for temporal)
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    if isinstance(data, list):
        data = np.asarray(data)

    if data.ndim == 3:
        # Temporal data (samples, seq_len, features)
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
    elif data.ndim == 2:
        # Static data (samples, features)
        min_val = np.min(data, axis=0)
        data = data - min_val
        max_val = np.max(data, axis=0)
        norm_data = data / (max_val + 1e-7)
    else:
        raise ValueError("Data must be 2D (static) or 3D (temporal)")

    return norm_data, min_val, max_val


# --- Keras Model Definitions ---
def _sequence_mask_lambda(t, maxlen):
    import tensorflow

    return tensorflow.sequence_mask(t, maxlen=maxlen)


def build_embedder(max_seq_len, dim_s, dim_x, hidden_dim, num_layers, module_name):
    """Builds the Embedder Keras Model."""
    X_S_input = Input(shape=(dim_s,), name="S_input")
    X_T_input = Input(shape=(max_seq_len, dim_x), name="X_input")
    T_input = Input(shape=(), dtype=tf.int32, name="T_input")

    mask = Lambda(
        _sequence_mask_lambda,
        arguments={"maxlen": max_seq_len},  # Pass max_seq_len as 'maxlen'
        output_shape=(max_seq_len,),
    )(T_input)

    # Static Embedder (e_S)
    H_S = Dense(hidden_dim, activation="sigmoid", name="Embedder_S_Dense")(X_S_input)

    # Temporal Embedder (e_X)
    # Condition the temporal embedder on the static latent code H_S
    H_S_repeated = RepeatVector(max_seq_len)(H_S)
    C_X = Concatenate(axis=-1, name="Embedder_X_Concat")([X_T_input, H_S_repeated])

    x = C_X
    for _ in range(num_layers):
        cell = rnn_cell(module_name, hidden_dim)
        x = RNN(cell, return_sequences=True)(x, mask=mask)
    e_outputs = x

    # Apply final Dense projection + sigmoid as in the original code
    H_T = Dense(hidden_dim, activation="sigmoid", name="Embedder_T_Dense")(e_outputs)

    H = [H_S, H_T]

    return Model(inputs=[X_S_input, X_T_input, T_input], outputs=H, name="embedder")


def build_recovery(max_seq_len, dim_s, dim_x, hidden_dim, num_layers, module_name):
    """Builds the Recovery Keras Model."""
    H_S_input = Input(shape=(hidden_dim,), name="H_S_input")
    H_T_input = Input(shape=(max_seq_len, hidden_dim), name="H_T_input")
    T_input = Input(shape=(), dtype=tf.int32, name="T_input")

    mask = Lambda(
        _sequence_mask_lambda,
        arguments={"maxlen": max_seq_len},
        output_shape=(max_seq_len,),
    )(T_input)

    # Static Recovery (r_S)
    X_S_tilde = Dense(dim_s, activation="sigmoid", name="Recovery_S_Dense")(H_S_input)

    # Temporal Recovery (r_X)
    x = H_T_input
    for _ in range(num_layers):
        cell = rnn_cell(module_name, hidden_dim)
        x = RNN(cell, return_sequences=True)(x, mask=mask)
    r_outputs = x

    X_T_tilde = Dense(dim_x, activation="sigmoid", name="Recovery_X_Dense")(r_outputs)

    X_tilde = [X_S_tilde, X_T_tilde]

    return Model(
        inputs=[H_S_input, H_T_input, T_input], outputs=X_tilde, name="recovery"
    )


def build_generator(max_seq_len, z_dim_s, z_dim_x, hidden_dim, num_layers, module_name):
    """Builds the Generator Keras Model."""
    Z_S_input = Input(shape=(z_dim_s,), name="Z_S_input")
    Z_T_input = Input(shape=(max_seq_len, z_dim_x), name="Z_T_input")
    T_input = Input(shape=(), dtype=tf.int32, name="T_input")

    mask = Lambda(
        _sequence_mask_lambda,
        arguments={"maxlen": max_seq_len},
        output_shape=(max_seq_len,),
    )(T_input)

    # Static Generator (g_S)
    E_S_hat = Dense(hidden_dim, activation="sigmoid", name="Generator_S_Dense")(
        Z_S_input
    )

    # Temporal Generator (g_X)
    # Condition the temporal generator on the static latent code H_S_hat
    E_S_hat_repeated = RepeatVector(max_seq_len)(E_S_hat)
    C_Z = Concatenate(axis=-1, name="Generator_X_Concat")([Z_T_input, E_S_hat_repeated])

    x = C_Z
    for _ in range(num_layers):
        cell = rnn_cell(module_name, hidden_dim)
        x = RNN(cell, return_sequences=True)(x, mask=mask)
    e_outputs = x  # This is the output from the RNN stack

    # Apply final Dense projection + sigmoid
    E_T_hat = tf.keras.layers.Dense(
        hidden_dim, activation="sigmoid", name="Generator_T_Dense"
    )(e_outputs)

    E_hat = [E_S_hat, E_T_hat]

    return Model(
        inputs=[Z_S_input, Z_T_input, T_input], outputs=E_hat, name="generator"
    )


def build_supervisor(max_seq_len, hidden_dim, num_layers, module_name):
    """Builds the Supervisor Keras Model."""
    H_S_input = Input(shape=(hidden_dim,), name="H_S_input_Sup")
    H_T_input = Input(shape=(max_seq_len, hidden_dim), name="H_T_input_Sup")
    T_input = Input(shape=(), dtype=tf.int32, name="T_input_Sup")

    mask = Lambda(
        _sequence_mask_lambda,
        arguments={"maxlen": max_seq_len},
        output_shape=(max_seq_len,),
    )(T_input)

    # Condition the supervisor on the static latent code H_S
    H_S_repeated = RepeatVector(max_seq_len)(H_S_input)
    C_H = Concatenate(axis=-1, name="Supervisor_Concat")([H_T_input, H_S_repeated])

    num_supervisor_layers = max(1, num_layers - 1)
    x = C_H
    for _ in range(num_supervisor_layers):
        cell = rnn_cell(module_name, hidden_dim)
        x = RNN(cell, return_sequences=True)(x, mask=mask)
    e_outputs = x

    S = Dense(hidden_dim, activation="sigmoid", name="Supervisor_Dense")(e_outputs)

    return Model(inputs=[H_S_input, H_T_input, T_input], outputs=S, name="supervisor")


def build_discriminator(max_seq_len, hidden_dim, num_layers, module_name):
    """Builds the Discriminator Keras Model."""
    H_S_input = Input(shape=(hidden_dim,), name="H_S_input_Disc")
    H_T_input = Input(shape=(max_seq_len, hidden_dim), name="H_T_input_Disc")
    T_input = Input(shape=(), dtype=tf.int32, name="T_input_Disc")

    mask = Lambda(
        _sequence_mask_lambda,
        arguments={"maxlen": max_seq_len},
        output_shape=(max_seq_len,),
    )(T_input)

    # Static Discriminator (d_S)
    Y_S_hat = Dense(1, activation=None, name="Discriminator_S_Dense")(H_S_input)

    # Temporal Discriminator (d_X)
    # Condition the temporal discriminator on the static latent code H_S
    H_S_repeated = RepeatVector(max_seq_len)(H_S_input)
    C_H = Concatenate(axis=-1, name="Discriminator_X_Concat")([H_T_input, H_S_repeated])

    x = C_H
    for _ in range(num_layers):
        cell = rnn_cell(module_name, hidden_dim)
        rnn_layer = RNN(cell, return_sequences=True)
        x = Bidirectional(rnn_layer, merge_mode="concat")(x, mask=mask)
    d_outputs = x

    Y_T_hat = Dense(1, activation=None, name="Discriminator_X_Dense")(d_outputs)

    Y_hat = [Y_S_hat, Y_T_hat]

    return Model(
        inputs=[H_S_input, H_T_input, T_input], outputs=Y_hat, name="discriminator"
    )


def timegan(ori_data_s, ori_data_x, parameters):
    """TimeGAN function (TensorFlow 2.x implementation).

    Args:
      - ori_data_s: original static time-series data
      - ori_data_x: original temporal time-series data
      - parameters: TimeGAN network parameters

    Returns:
      - generated_data_s: generated static time-series data
      - generated_data_x: generated temporal time-series data
    """

    if isinstance(ori_data_s, list):
        ori_data_s = np.asarray(ori_data_s)
    if isinstance(ori_data_x, list):
        ori_data_x = np.asarray(ori_data_x)

    # Data parameters
    no, seq_len, dim_x = ori_data_x.shape
    dim_s = ori_data_s.shape[1]
    ori_time, max_seq_len = extract_time(ori_data_x)

    # Normalize data
    ori_data_s, min_val_s, max_val_s = MinMaxScaler(ori_data_s)
    ori_data_x, min_val_x, max_val_x = MinMaxScaler(ori_data_x)

    # Network parameters
    hidden_dim = parameters["hidden_dim"]
    num_layers = parameters["num_layer"]
    iterations = parameters["iterations"]
    batch_size = parameters["batch_size"]
    module_name = parameters["module"]
    z_dim_s = dim_s  # Use static feature dim for static z_dim
    z_dim_x = dim_x  # Use temporal feature dim for temporal z_dim
    gamma = 1

    # Build Models
    embedder = build_embedder(
        max_seq_len, dim_s, dim_x, hidden_dim, num_layers, module_name
    )
    recovery = build_recovery(
        max_seq_len, dim_s, dim_x, hidden_dim, num_layers, module_name
    )
    generator = build_generator(
        max_seq_len, z_dim_s, z_dim_x, hidden_dim, num_layers, module_name
    )
    supervisor = build_supervisor(max_seq_len, hidden_dim, num_layers, module_name)
    discriminator = build_discriminator(
        max_seq_len, hidden_dim, num_layers, module_name
    )

    # Loss & Optimizers
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()

    E_optimizer = tf.keras.optimizers.Adam()
    D_optimizer = tf.keras.optimizers.Adam()
    G_optimizer = tf.keras.optimizers.Adam()
    GS_optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step_embedder(S_mb, X_mb, T_mb):
        with tf.GradientTape() as tape:
            [H_S, H_T] = embedder([S_mb, X_mb, T_mb], training=True)
            [X_S_tilde, X_T_tilde] = recovery([H_S, H_T, T_mb], training=True)

            E_loss_S = mse(S_mb, X_S_tilde)
            E_loss_T = mse(X_mb, X_T_tilde)
            E_loss_T0 = E_loss_S + E_loss_T
            E_loss0 = 10 * tf.sqrt(E_loss_T0)

        vars_e = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(E_loss0, vars_e)
        E_optimizer.apply_gradients(zip(gradients, vars_e))
        return E_loss_T0

    @tf.function
    def train_step_supervisor(S_mb, X_mb, T_mb):
        with tf.GradientTape() as tape:
            [H_S, H_T] = embedder([S_mb, X_mb, T_mb], training=True)
            H_hat_supervise = supervisor([H_S, H_T, T_mb], training=True)
            # Supervised loss is only on temporal dynamics
            G_loss_S = mse(H_T[:, 1:, :], H_hat_supervise[:, :-1, :])

        vars_gs = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(G_loss_S, vars_gs)
        GS_optimizer.apply_gradients(zip(gradients, vars_gs))
        return G_loss_S

    @tf.function
    def train_step_joint(S_mb, X_mb, T_mb, Z_S_mb, Z_T_mb):
        # --- Generator Training Twice---
        with tf.GradientTape() as g_tape:
            # Embed real data
            [H_S, H_T] = embedder([S_mb, X_mb, T_mb], training=True)
            # Generate fake latents
            [E_S_hat, E_T_hat] = generator([Z_S_mb, Z_T_mb, T_mb], training=True)
            # Supervise fake latents
            H_T_hat = supervisor([E_S_hat, E_T_hat, T_mb], training=True)
            # Reconstruct fake data
            [X_S_hat, X_T_hat] = recovery([E_S_hat, H_T_hat, T_mb], training=True)

            # Discriminate
            [Y_S_fake, Y_T_fake] = discriminator(
                [E_S_hat, H_T_hat, T_mb], training=False
            )
            [Y_S_fake_e, Y_T_fake_e] = discriminator(
                [E_S_hat, E_T_hat, T_mb], training=False
            )

            # --- G_loss ---
            # 1. Adversarial Loss
            G_loss_U_S = bce(tf.ones_like(Y_S_fake), Y_S_fake)
            G_loss_U_T = bce(tf.ones_like(Y_T_fake), Y_T_fake)
            G_loss_U = G_loss_U_S + G_loss_U_T

            G_loss_U_e_S = bce(tf.ones_like(Y_S_fake_e), Y_S_fake_e)
            G_loss_U_e_T = bce(tf.ones_like(Y_T_fake_e), Y_T_fake_e)
            G_loss_U_e = G_loss_U_e_S + G_loss_U_e_T

            # 2. Supervised Loss
            H_hat_supervise = supervisor([H_S, H_T, T_mb], training=True)
            G_loss_S = mse(H_T[:, 1:, :], H_hat_supervise[:, :-1, :])

            # 3. Moment Loss
            G_loss_V_S1 = tf.reduce_mean(
                tf.abs(
                    tf.sqrt(tf.nn.moments(X_S_hat, [0])[1] + 1e-6)
                    - tf.sqrt(tf.nn.moments(S_mb, [0])[1] + 1e-6)
                )
            )
            G_loss_V_S2 = tf.reduce_mean(
                tf.abs((tf.nn.moments(X_S_hat, [0])[0]) - (tf.nn.moments(S_mb, [0])[0]))
            )
            G_loss_V_S = G_loss_V_S1 + G_loss_V_S2

            G_loss_V_T1 = tf.reduce_mean(
                tf.abs(
                    tf.sqrt(tf.nn.moments(X_T_hat, [0, 1])[1] + 1e-6)
                    - tf.sqrt(tf.nn.moments(X_mb, [0, 1])[1] + 1e-6)
                )
            )
            G_loss_V_T2 = tf.reduce_mean(
                tf.abs(
                    (tf.nn.moments(X_T_hat, [0, 1])[0])
                    - (tf.nn.moments(X_mb, [0, 1])[0])
                )
            )
            G_loss_V_T = G_loss_V_T1 + G_loss_V_T2
            G_loss_V = G_loss_V_S + G_loss_V_T

            # Total G_loss
            G_loss = (
                G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V
            )

        vars_g = generator.trainable_variables + supervisor.trainable_variables
        gradients_g = g_tape.gradient(G_loss, vars_g)
        G_optimizer.apply_gradients(zip(gradients_g, vars_g))

        # --- Generator Training Again ---
        # (This is a repeat from the original code, kept for consistency)
        with tf.GradientTape() as g_tape:
            [H_S, H_T] = embedder([S_mb, X_mb, T_mb], training=True)
            [E_S_hat, E_T_hat] = generator([Z_S_mb, Z_T_mb, T_mb], training=True)
            H_T_hat = supervisor([E_S_hat, E_T_hat, T_mb], training=True)
            [X_S_hat, X_T_hat] = recovery([E_S_hat, H_T_hat, T_mb], training=True)
            [Y_S_fake, Y_T_fake] = discriminator(
                [E_S_hat, H_T_hat, T_mb], training=False
            )
            [Y_S_fake_e, Y_T_fake_e] = discriminator(
                [E_S_hat, E_T_hat, T_mb], training=False
            )

            G_loss_U_S = bce(tf.ones_like(Y_S_fake), Y_S_fake)
            G_loss_U_T = bce(tf.ones_like(Y_T_fake), Y_T_fake)
            G_loss_U = G_loss_U_S + G_loss_U_T
            G_loss_U_e_S = bce(tf.ones_like(Y_S_fake_e), Y_S_fake_e)
            G_loss_U_e_T = bce(tf.ones_like(Y_T_fake_e), Y_T_fake_e)
            G_loss_U_e = G_loss_U_e_S + G_loss_U_e_T
            H_hat_supervise = supervisor([H_S, H_T, T_mb], training=True)
            G_loss_S = mse(H_T[:, 1:, :], H_hat_supervise[:, :-1, :])
            G_loss_V_S1 = tf.reduce_mean(
                tf.abs(
                    tf.sqrt(tf.nn.moments(X_S_hat, [0])[1] + 1e-6)
                    - tf.sqrt(tf.nn.moments(S_mb, [0])[1] + 1e-6)
                )
            )
            G_loss_V_S2 = tf.reduce_mean(
                tf.abs((tf.nn.moments(X_S_hat, [0])[0]) - (tf.nn.moments(S_mb, [0])[0]))
            )
            G_loss_V_S = G_loss_V_S1 + G_loss_V_S2
            G_loss_V_T1 = tf.reduce_mean(
                tf.abs(
                    tf.sqrt(tf.nn.moments(X_T_hat, [0, 1])[1] + 1e-6)
                    - tf.sqrt(tf.nn.moments(X_mb, [0, 1])[1] + 1e-6)
                )
            )
            G_loss_V_T2 = tf.reduce_mean(
                tf.abs(
                    (tf.nn.moments(X_T_hat, [0, 1])[0])
                    - (tf.nn.moments(X_mb, [0, 1])[0])
                )
            )
            G_loss_V_T = G_loss_V_T1 + G_loss_V_T2
            G_loss_V = G_loss_V_S + G_loss_V_T
            G_loss = (
                G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V
            )

        vars_g = generator.trainable_variables + supervisor.trainable_variables
        gradients_g = g_tape.gradient(G_loss, vars_g)
        G_optimizer.apply_gradients(zip(gradients_g, vars_g))

        # --- Embedder Training Twice---
        with tf.GradientTape() as e_tape:
            [H_S, H_T] = embedder([S_mb, X_mb, T_mb], training=True)
            [S_tilde, X_tilde] = recovery([H_S, H_T, T_mb], training=True)
            H_hat_supervise = supervisor([H_S, H_T, T_mb], training=True)

            G_loss_S_e = mse(H_T[:, 1:, :], H_hat_supervise[:, :-1, :])
            E_loss_S = mse(S_mb, S_tilde)
            E_loss_T = mse(X_mb, X_tilde)
            E_loss_T0 = E_loss_S + E_loss_T
            E_loss0 = 10 * tf.sqrt(E_loss_T0)
            E_loss = E_loss0 + 0.1 * G_loss_S_e

        vars_e = embedder.trainable_variables + recovery.trainable_variables
        gradients_e = e_tape.gradient(E_loss, vars_e)
        E_optimizer.apply_gradients(zip(gradients_e, vars_e))

        with tf.GradientTape() as e_tape:
            [H_S, H_T] = embedder([S_mb, X_mb, T_mb], training=True)
            [S_tilde, X_tilde] = recovery([H_S, H_T, T_mb], training=True)
            H_hat_supervise = supervisor([H_S, H_T, T_mb], training=True)

            G_loss_S_e = mse(H_T[:, 1:, :], H_hat_supervise[:, :-1, :])
            E_loss_S = mse(S_mb, S_tilde)
            E_loss_T = mse(X_mb, X_tilde)
            E_loss_T0 = E_loss_S + E_loss_T
            E_loss0 = 10 * tf.sqrt(E_loss_T0)
            E_loss = E_loss0 + 0.1 * G_loss_S_e

        vars_e = embedder.trainable_variables + recovery.trainable_variables
        gradients_e = e_tape.gradient(E_loss, vars_e)
        E_optimizer.apply_gradients(zip(gradients_e, vars_e))

        # --- Discriminator Training Once---
        with tf.GradientTape() as d_tape:
            [H_S, H_T] = embedder([S_mb, X_mb, T_mb], training=False)
            [E_S_hat, E_T_hat] = generator([Z_S_mb, Z_T_mb, T_mb], training=False)
            H_T_hat = supervisor([E_S_hat, E_T_hat, T_mb], training=False)

            [Y_S_fake, Y_T_fake] = discriminator(
                [E_S_hat, H_T_hat, T_mb], training=True
            )
            [Y_S_real, Y_T_real] = discriminator([H_S, H_T, T_mb], training=True)
            [Y_S_fake_e, Y_T_fake_e] = discriminator(
                [E_S_hat, E_T_hat, T_mb], training=True
            )

            D_loss_real = bce(tf.ones_like(Y_S_real), Y_S_real) + bce(
                tf.ones_like(Y_T_real), Y_T_real
            )
            D_loss_fake = bce(tf.zeros_like(Y_S_fake), Y_S_fake) + bce(
                tf.zeros_like(Y_T_fake), Y_T_fake
            )
            D_loss_fake_e = bce(tf.zeros_like(Y_S_fake_e), Y_S_fake_e) + bce(
                tf.zeros_like(Y_T_fake_e), Y_T_fake_e
            )
            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        if D_loss > 0.15:
            vars_d = discriminator.trainable_variables
            gradients_d = d_tape.gradient(D_loss, vars_d)
            D_optimizer.apply_gradients(zip(gradients_d, vars_d))

        return D_loss, G_loss_U, G_loss_S, G_loss_V, E_loss_T0

    # 1. Embedding network training
    print("Start Embedding Network Training")
    for itt in range(iterations):
        S_mb, X_mb, T_mb = batch_generator(ori_data_s, ori_data_x, ori_time, batch_size)
        S_mb_t = tf.convert_to_tensor(S_mb, dtype=tf.float32)
        X_mb_t = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        T_mb_t = tf.convert_to_tensor(T_mb, dtype=tf.int32)

        step_e_loss = train_step_embedder(S_mb_t, X_mb_t, T_mb_t)
        if itt % 1000 == 0:
            print(
                f"step: {itt}/{iterations}, e_loss: {np.round(np.sqrt(step_e_loss), 4)}"
            )

    print("Finish Embedding Network Training")

    # 2. Training only with supervised loss
    print("Start Training with Supervised Loss Only")
    for itt in range(iterations):
        S_mb, X_mb, T_mb = batch_generator(ori_data_s, ori_data_x, ori_time, batch_size)
        S_mb_t = tf.convert_to_tensor(S_mb, dtype=tf.float32)
        X_mb_t = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        T_mb_t = tf.convert_to_tensor(T_mb, dtype=tf.int32)

        step_g_loss_s = train_step_supervisor(S_mb_t, X_mb_t, T_mb_t)
        if itt % 1000 == 0:
            print(
                f"step: {itt}/{iterations}, s_loss: {np.round(np.sqrt(step_g_loss_s), 4)}"
            )

    print("Finish Training with Supervised Loss Only")

    # 3. Joint Training
    print("Start Joint Training")
    step_d_loss = 0.0
    for itt in range(iterations):
        S_mb, X_mb, T_mb = batch_generator(ori_data_s, ori_data_x, ori_time, batch_size)
        Z_S_mb, Z_T_mb = random_generator(
            batch_size, z_dim_s, z_dim_x, T_mb, max_seq_len
        )

        S_mb_t = tf.convert_to_tensor(S_mb, dtype=tf.float32)
        X_mb_t = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        T_mb_t = tf.convert_to_tensor(T_mb, dtype=tf.int32)
        Z_S_mb_t = tf.convert_to_tensor(Z_S_mb, dtype=tf.float32)
        Z_T_mb_t = tf.convert_to_tensor(Z_T_mb, dtype=tf.float32)

        step_d_loss, step_g_loss_u, step_g_loss_s, step_g_loss_v, step_e_loss_t0 = (
            train_step_joint(S_mb_t, X_mb_t, T_mb_t, Z_S_mb_t, Z_T_mb_t)
        )

        if itt % 1000 == 0 or itt == iterations - 1:
            print(
                f"step: {itt}/{iterations}, "
                f"d_loss: {np.round(step_d_loss.numpy(), 4)}, "
                f"g_loss_u: {np.round(step_g_loss_u.numpy(), 4)}, "
                f"g_loss_s: {np.round(np.sqrt(step_g_loss_s.numpy()), 4)}, "
                f"g_loss_v: {np.round(step_g_loss_v.numpy(), 4)}, "
                f"e_loss_t0: {np.round(np.sqrt(step_e_loss_t0.numpy()), 4)}"
            )

    print("Finish Joint Training")

    # ------------------------------------------------------------------
    # --- SAVE TRAINED MODELS AND SCALERS ---
    # ------------------------------------------------------------------
    print("Saving trained models and scalers...")

    # Define file paths
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)  # Add 'import os' at the top of timegan.py

    gen_path = os.path.join(save_dir, "generator.h5")
    sup_path = os.path.join(save_dir, "supervisor.h5")
    rec_path = os.path.join(save_dir, "recovery.h5")
    scaler_path = os.path.join(save_dir, "scalers.npz")

    # Save models
    # Note: Using .h5 format for simplicity here.
    generator.save(gen_path)
    supervisor.save(sup_path)
    recovery.save(rec_path)

    # Save scalers
    np.savez(
        scaler_path,
        min_val_s=min_val_s,
        max_val_s=max_val_s,
        min_val_x=min_val_x,
        max_val_x=max_val_x,
    )
    print(f"Models and scalers saved to {save_dir}/")
    # ------------------------------------------------------------------


def generate_data_from_saved_models(
    ori_data_s,
    ori_data_x,
    save_dir="saved_models",
):
    """
    Generates synthetic data using pre-trained models.

    Args:
        - ori_data_s: original static data (for shape/param info)
        - ori_data_x: original temporal data (for shape/param info)
        - save_dir: directory where models and scalers are saved.

    Returns:
        - generated_data_s: generated static time-series data
        - generated_data_x: generated temporal time-series data
    """

    print("Loading pre-trained models and scalers...")

    # --- 1. Define Paths and Load Scalers ---
    gen_path = os.path.join(save_dir, "generator.h5")
    sup_path = os.path.join(save_dir, "supervisor.h5")
    rec_path = os.path.join(save_dir, "recovery.h5")
    scaler_path = os.path.join(save_dir, "scalers.npz")

    try:
        scalers = np.load(scaler_path)
        min_val_s = scalers["min_val_s"]
        max_val_s = scalers["max_val_s"]
        min_val_x = scalers["min_val_x"]
        max_val_x = scalers["max_val_x"]
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {scaler_path}")
        return

    # --- 2. Load Models with Custom Object ---
    # We must provide LayerNormLSTMCell as a custom object
    # since it's a custom class used in the models.
    custom_objects = {
        "LayerNormLSTMCell": LayerNormLSTMCell,
        "_sequence_mask_lambda": _sequence_mask_lambda,
    }

    try:
        generator = tf.keras.models.load_model(
            gen_path, custom_objects=custom_objects, safe_mode=False
        )
        supervisor = tf.keras.models.load_model(
            sup_path, custom_objects=custom_objects, safe_mode=False
        )
        recovery = tf.keras.models.load_model(
            rec_path, custom_objects=custom_objects, safe_mode=False
        )
    except (IOError, FileNotFoundError) as e:
        print(f"Error loading models from {save_dir}. Did you train them first?")
        print(e)
        return

    print("Models loaded successfully.")

    # --- 3. Get Data Parameters ---
    if isinstance(ori_data_s, list):
        ori_data_s = np.asarray(ori_data_s)
    if isinstance(ori_data_x, list):
        ori_data_x = np.asarray(ori_data_x)

    no, seq_len, dim_x = ori_data_x.shape
    dim_s = ori_data_s.shape[1]
    ori_time, max_seq_len = extract_time(ori_data_x)
    z_dim_s = dim_s
    z_dim_x = dim_x

    # --- 4. Generate Synthetic Data (Copied from timegan function) ---
    print("Generating synthetic data...")
    Z_S_mb, Z_T_mb = random_generator(no, z_dim_s, z_dim_x, ori_time, max_seq_len)
    Z_S_mb_t = tf.convert_to_tensor(Z_S_mb, dtype=tf.float32)
    Z_T_mb_t = tf.convert_to_tensor(Z_T_mb, dtype=tf.float32)
    T_mb_t = tf.convert_to_tensor(ori_time, dtype=tf.int32)

    [E_S_hat, E_T_hat] = generator([Z_S_mb_t, Z_T_mb_t, T_mb_t], training=False)
    H_T_hat = supervisor([E_S_hat, E_T_hat, T_mb_t], training=False)
    [X_S_hat, X_T_hat_curr] = recovery([E_S_hat, H_T_hat, T_mb_t], training=False)

    generated_data_s = X_S_hat.numpy()
    generated_data_x_curr = X_T_hat_curr.numpy()

    generated_data_x = list()
    for i in range(no):
        temp = generated_data_x_curr[i, : ori_time[i], :]
        generated_data_x.append(temp)

    # --- 5. Renormalize Data ---
    generated_data_s = generated_data_s * (max_val_s + 1e-7) + min_val_s
    generated_data_x = np.asarray(generated_data_x, dtype=object)
    generated_data_x = generated_data_x * (max_val_x + 1e-7) + min_val_x

    print("Data generation complete.")
    return list(generated_data_s), list(generated_data_x)

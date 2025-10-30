import tensorflow as tf
import numpy as np

# Assuming the 'utils' functions (rnn_cell, etc.) have been
# updated for TF 2.x as we did in previous steps.
from utils import extract_time, rnn_cell, random_generator, batch_generator


# Helper function (from original code, requires no changes)
def MinMaxScaler(data):
    """Min-Max Normalizer.
    Args:
      - data: raw data
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    if isinstance(data, list):
        data = np.asarray(data)

    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val


# --- Keras Model Definitions (Replaces tf.variable_scope) ---
# This version stacks RNN layers in a loop, which is the
# correct and robust way to build a MultiRNN in TF 2.x.


def build_embedder(max_seq_len, dim, hidden_dim, num_layers, module_name):
    """Builds the Embedder Keras Model."""
    X_input = tf.keras.Input(shape=(max_seq_len, dim), name="X_input")
    T_input = tf.keras.Input(shape=(), dtype=tf.int32, name="T_input")

    mask = tf.keras.layers.Lambda(lambda t: tf.sequence_mask(t, maxlen=max_seq_len))(
        T_input
    )

    # --- FIX APPLIED HERE ---
    # Stack RNN layers in a loop
    x = X_input
    for _ in range(num_layers):
        cell = rnn_cell(module_name, hidden_dim)
        # return_sequences=True is required for all but the last layer
        # in a stack, but here we need it for all layers.
        x = tf.keras.layers.RNN(cell, return_sequences=True)(x, mask=mask)
    e_outputs = x
    # --- END OF FIX ---

    H = tf.keras.layers.Dense(hidden_dim, activation="sigmoid", name="Embedder_Dense")(
        e_outputs
    )

    return tf.keras.Model(inputs=[X_input, T_input], outputs=H, name="embedder")


def build_recovery(max_seq_len, dim, hidden_dim, num_layers, module_name):
    """Builds the Recovery Keras Model."""
    H_input = tf.keras.Input(shape=(max_seq_len, hidden_dim), name="H_input")
    T_input = tf.keras.Input(shape=(), dtype=tf.int32, name="T_input")

    mask = tf.keras.layers.Lambda(lambda t: tf.sequence_mask(t, maxlen=max_seq_len))(
        T_input
    )

    # --- FIX APPLIED HERE ---
    x = H_input
    for _ in range(num_layers):
        cell = rnn_cell(module_name, hidden_dim)
        x = tf.keras.layers.RNN(cell, return_sequences=True)(x, mask=mask)
    r_outputs = x
    # --- END OF FIX ---

    X_tilde = tf.keras.layers.Dense(dim, activation="sigmoid", name="Recovery_Dense")(
        r_outputs
    )

    return tf.keras.Model(inputs=[H_input, T_input], outputs=X_tilde, name="recovery")


def build_generator(max_seq_len, z_dim, hidden_dim, num_layers, module_name):
    """Builds the Generator Keras Model."""
    Z_input = tf.keras.Input(shape=(max_seq_len, z_dim), name="Z_input")
    T_input = tf.keras.Input(shape=(), dtype=tf.int32, name="T_input")

    mask = tf.keras.layers.Lambda(lambda t: tf.sequence_mask(t, maxlen=max_seq_len))(
        T_input
    )

    # --- FIX APPLIED HERE ---
    x = Z_input
    for _ in range(num_layers):
        cell = rnn_cell(module_name, hidden_dim)
        x = tf.keras.layers.RNN(cell, return_sequences=True)(x, mask=mask)
    e_outputs = x
    # --- END OF FIX ---

    E = tf.keras.layers.Dense(hidden_dim, activation="sigmoid", name="Generator_Dense")(
        e_outputs
    )

    return tf.keras.Model(inputs=[Z_input, T_input], outputs=E, name="generator")


def build_supervisor(max_seq_len, hidden_dim, num_layers, module_name):
    """Builds the Supervisor Keras Model."""
    H_input = tf.keras.Input(shape=(max_seq_len, hidden_dim), name="H_input_Sup")
    T_input = tf.keras.Input(shape=(), dtype=tf.int32, name="T_input_Sup")

    mask = tf.keras.layers.Lambda(lambda t: tf.sequence_mask(t, maxlen=max_seq_len))(
        T_input
    )

    # --- FIX APPLIED HERE ---
    # Ensure at least one layer, as original code implies num_layers >= 2
    num_supervisor_layers = max(1, num_layers - 1)

    x = H_input
    for _ in range(num_supervisor_layers):
        cell = rnn_cell(module_name, hidden_dim)
        x = tf.keras.layers.RNN(cell, return_sequences=True)(x, mask=mask)
    e_outputs = x
    # --- END OF FIX ---

    S = tf.keras.layers.Dense(
        hidden_dim, activation="sigmoid", name="Supervisor_Dense"
    )(e_outputs)

    return tf.keras.Model(inputs=[H_input, T_input], outputs=S, name="supervisor")


def build_discriminator(max_seq_len, hidden_dim, num_layers, module_name):
    """Builds the Discriminator Keras Model."""
    H_input = tf.keras.Input(shape=(max_seq_len, hidden_dim), name="H_input_Disc")
    T_input = tf.keras.Input(shape=(), dtype=tf.int32, name="T_input_Disc")

    mask = tf.keras.layers.Lambda(lambda t: tf.sequence_mask(t, maxlen=max_seq_len))(
        T_input
    )

    # --- FIX APPLIED HERE ---
    x = H_input
    for _ in range(num_layers):
        cell = rnn_cell(module_name, hidden_dim)
        x = tf.keras.layers.RNN(cell, return_sequences=True)(x, mask=mask)
    d_outputs = x
    # --- END OF FIX ---

    Y_hat = tf.keras.layers.Dense(1, activation=None, name="Discriminator_Dense")(
        d_outputs
    )

    return tf.keras.Model(
        inputs=[H_input, T_input], outputs=Y_hat, name="discriminator"
    )


def timegan(ori_data, parameters):
    """TimeGAN function (TensorFlow 2.x implementation).

    Args:
      - ori_data: original time-series data
      - parameters: TimeGAN network parameters

    Returns:
      - generated_data: generated time-series data
    """

    if isinstance(ori_data, list):
        ori_data = np.asarray(ori_data)

    no, seq_len, dim = ori_data.shape
    ori_time, max_seq_len = extract_time(ori_data)
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    hidden_dim = parameters["hidden_dim"]
    num_layers = parameters["num_layer"]
    iterations = parameters["iterations"]
    batch_size = parameters["batch_size"]
    module_name = parameters["module"]
    z_dim = dim
    gamma = 1

    embedder = build_embedder(max_seq_len, dim, hidden_dim, num_layers, module_name)
    recovery = build_recovery(max_seq_len, dim, hidden_dim, num_layers, module_name)
    generator = build_generator(max_seq_len, z_dim, hidden_dim, num_layers, module_name)
    supervisor = build_supervisor(max_seq_len, hidden_dim, num_layers, module_name)
    discriminator = build_discriminator(
        max_seq_len, hidden_dim, num_layers, module_name
    )

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()

    E_optimizer = tf.keras.optimizers.Adam()
    D_optimizer = tf.keras.optimizers.Adam()
    G_optimizer = tf.keras.optimizers.Adam()
    GS_optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step_embedder(X_mb, T_mb):
        with tf.GradientTape() as tape:
            H = embedder([X_mb, T_mb], training=True)
            X_tilde = recovery([H, T_mb], training=True)
            E_loss_T0 = mse(X_mb, X_tilde)
            E_loss0 = 10 * tf.sqrt(E_loss_T0)

        vars_e = embedder.trainable_variables + recovery.trainable_variables
        gradients = tape.gradient(E_loss0, vars_e)
        E_optimizer.apply_gradients(zip(gradients, vars_e))
        return E_loss_T0

    @tf.function
    def train_step_supervisor(X_mb, T_mb):
        with tf.GradientTape() as tape:
            H = embedder([X_mb, T_mb], training=True)
            H_hat_supervise = supervisor([H, T_mb], training=True)
            G_loss_S = mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])

        vars_gs = generator.trainable_variables + supervisor.trainable_variables
        gradients = tape.gradient(G_loss_S, vars_gs)
        GS_optimizer.apply_gradients(zip(gradients, vars_gs))
        return G_loss_S

    @tf.function
    def train_step_joint(X_mb, T_mb, Z_mb):
        # --- Generator Training ---
        with tf.GradientTape() as g_tape:
            H = embedder([X_mb, T_mb], training=True)
            E_hat = generator([Z_mb, T_mb], training=True)
            H_hat = supervisor([E_hat, T_mb], training=True)
            X_hat = recovery([H_hat, T_mb], training=True)

            Y_fake = discriminator([H_hat, T_mb], training=False)
            Y_real = discriminator([H, T_mb], training=False)
            Y_fake_e = discriminator([E_hat, T_mb], training=False)

            G_loss_U = bce(tf.ones_like(Y_fake), Y_fake)
            G_loss_U_e = bce(tf.ones_like(Y_fake_e), Y_fake_e)

            H_hat_supervise = supervisor([H, T_mb], training=True)
            G_loss_S = mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])

            G_loss_V1 = tf.reduce_mean(
                tf.abs(
                    tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6)
                    - tf.sqrt(tf.nn.moments(X_mb, [0])[1] + 1e-6)
                )
            )
            G_loss_V2 = tf.reduce_mean(
                tf.abs((tf.nn.moments(X_hat, [0])[0]) - (tf.nn.moments(X_mb, [0])[0]))
            )
            G_loss_V = G_loss_V1 + G_loss_V2

            G_loss = (
                G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V
            )

        vars_g = generator.trainable_variables + supervisor.trainable_variables
        gradients_g = g_tape.gradient(G_loss, vars_g)
        G_optimizer.apply_gradients(zip(gradients_g, vars_g))

        # --- Embedder Training ---
        with tf.GradientTape() as e_tape:
            H = embedder([X_mb, T_mb], training=True)
            X_tilde = recovery([H, T_mb], training=True)

            H_hat_supervise = supervisor([H, T_mb], training=True)
            G_loss_S_e = mse(
                H[:, 1:, :], H_hat_supervise[:, :-1, :]
            )  # Use a different name

            E_loss_T0 = mse(X_mb, X_tilde)

            E_loss0 = 10 * tf.sqrt(E_loss_T0)
            E_loss = E_loss0 + 0.1 * G_loss_S_e

        vars_e = embedder.trainable_variables + recovery.trainable_variables
        gradients_e = e_tape.gradient(E_loss, vars_e)
        E_optimizer.apply_gradients(zip(gradients_e, vars_e))

        # --- Discriminator Training ---
        with tf.GradientTape() as d_tape:
            H = embedder([X_mb, T_mb], training=False)
            E_hat = generator([Z_mb, T_mb], training=False)
            H_hat = supervisor([E_hat, T_mb], training=False)

            Y_fake = discriminator([H_hat, T_mb], training=True)
            Y_real = discriminator([H, T_mb], training=True)
            Y_fake_e = discriminator([E_hat, T_mb], training=True)

            D_loss_real = bce(tf.ones_like(Y_real), Y_real)
            D_loss_fake = bce(tf.zeros_like(Y_fake), Y_fake)
            D_loss_fake_e = bce(tf.zeros_like(Y_fake_e), Y_fake_e)
            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        if D_loss > 0.15:
            vars_d = discriminator.trainable_variables
            gradients_d = d_tape.gradient(D_loss, vars_d)
            D_optimizer.apply_gradients(zip(gradients_d, vars_d))

        return D_loss, G_loss_U, G_loss_S, G_loss_V, E_loss_T0

    # 1. Embedding network training
    print("Start Embedding Network Training")
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb_t = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        T_mb_t = tf.convert_to_tensor(T_mb, dtype=tf.int32)
        step_e_loss = train_step_embedder(X_mb_t, T_mb_t)
        if itt % 1000 == 0:
            print(
                f"step: {itt}/{iterations}, e_loss: {np.round(np.sqrt(step_e_loss), 4)}"
            )

    print("Finish Embedding Network Training")

    # 2. Training only with supervised loss
    print("Start Training with Supervised Loss Only")
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb_t = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        T_mb_t = tf.convert_to_tensor(T_mb, dtype=tf.int32)
        step_g_loss_s = train_step_supervisor(X_mb_t, T_mb_t)
        if itt % 1000 == 0:
            print(
                f"step: {itt}/{iterations}, s_loss: {np.round(np.sqrt(step_g_loss_s), 4)}"
            )

    print("Finish Training with Supervised Loss Only")

    # 3. Joint Training
    print("Start Joint Training")
    step_d_loss = 0.0  # Initialize in case d_loss isn't run
    for itt in range(iterations):
        for _ in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
            X_mb_t = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            T_mb_t = tf.convert_to_tensor(T_mb, dtype=tf.int32)
            Z_mb_t = tf.convert_to_tensor(Z_mb, dtype=tf.float32)
            _, step_g_loss_u, step_g_loss_s, step_g_loss_v, step_e_loss_t0 = (
                train_step_joint(X_mb_t, T_mb_t, Z_mb_t)
            )

        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        X_mb_t = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        T_mb_t = tf.convert_to_tensor(T_mb, dtype=tf.int32)
        Z_mb_t = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

        step_d_loss, step_g_loss_u, step_g_loss_s, step_g_loss_v, step_e_loss_t0 = (
            train_step_joint(X_mb_t, T_mb_t, Z_mb_t)
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

    ## Synthetic data generation
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    Z_mb_t = tf.convert_to_tensor(Z_mb, dtype=tf.float32)
    T_mb_t = tf.convert_to_tensor(ori_time, dtype=tf.int32)

    E_hat = generator([Z_mb_t, T_mb_t], training=False)
    H_hat = supervisor([E_hat, T_mb_t], training=False)
    generated_data_curr = recovery([H_hat, T_mb_t], training=False)

    generated_data_curr = generated_data_curr.numpy()

    generated_data = list()
    for i in range(no):
        temp = generated_data_curr[i, : ori_time[i], :]
        generated_data.append(temp)

    generated_data = np.asarray(generated_data, dtype=object)
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    return list(generated_data)

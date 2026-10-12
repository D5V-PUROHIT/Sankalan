import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import json
import os

class VAE(Model):
    def __init__(self, input_dim, latent_dim, feature_map, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.feature_map = feature_map
        
        # Encoder
        self.encoder_inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(128, activation="relu")(self.encoder_inputs)
        x = layers.Dense(64, activation="relu")(x)
        self.z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        
        # Sampling layer
        self.z = Sampling()([self.z_mean, self.z_log_var])
        self.encoder = Model(self.encoder_inputs, [self.z_mean, self.z_log_var, self.z], name="encoder")
        
        # Decoder
        self.latent_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(64, activation="relu")(self.latent_inputs)
        x = layers.Dense(128, activation="relu")(x)
        self.decoder_outputs = layers.Dense(input_dim, name="decoder_output")(x)
        self.decoder = Model(self.latent_inputs, self.decoder_outputs, name="decoder")

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a dataset row."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Custom Loss Functions (Addendum B)
def compute_reconstruction_loss(x_true, x_recon, feature_map):
    """
    Computes MSE for numeric features and Categorical Crossentropy for One-Hot features
    based on the feature_map slices.
    """
    x_true = tf.cast(x_true, tf.float32)
    x_recon = tf.cast(x_recon, tf.float32)
    loss = 0.0
    for feature in feature_map:
        start = feature["start"]
        end = feature["end"]
        ftype = feature["type"]
        
        true_slice = x_true[:, start:end]
        recon_slice = x_recon[:, start:end]
        
        if ftype == "numeric":
            loss += tf.reduce_mean(tf.square(true_slice - recon_slice), axis=-1)
        else:  # categorical
            # Apply softmax to the reconstructed logits for this categorical block
            recon_probs = tf.nn.softmax(recon_slice, axis=-1)
            # Clip probabilities to avoid log(0)
            recon_probs = tf.clip_by_value(recon_probs, 1e-7, 1.0 - 1e-7)
            # Compute cross entropy
            ce = -tf.reduce_sum(true_slice * tf.math.log(recon_probs), axis=-1)
            loss += ce
            
    return tf.reduce_mean(loss)

def get_beta(epoch, total_epochs, target_beta=1.0, warmup_frac=0.25):
    """Warmup scheduler for KL Divergence Beta."""
    warmup_epochs = total_epochs * warmup_frac
    if epoch >= warmup_epochs:
        return target_beta
    return target_beta * (epoch / max(1.0, warmup_epochs))

def train_step(model, data, optimizer, feature_map, beta):
    data = tf.cast(data, tf.float32)
    beta = tf.cast(beta, tf.float32)
    with tf.GradientTape() as tape:
        reconstructed, z_mean, z_log_var = model(data)
        
        # 1. Reconstruction Loss (Composite)
        recon_loss = compute_reconstruction_loss(data, reconstructed, feature_map)
        
        # 2. KL Divergence
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        
        # 3. Total Loss
        total_loss = recon_loss + (beta * kl_loss)
        
    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    return {"total_loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

def train_vae(X_train, feature_map, latent_dim=8, epochs=50, batch_size=32, progress_callback=None, check_stop_callback=None):
    """
    Sets up the dataset, optimizer, and training loop with progress tracking and stop support.
    Returns the trained model and training history.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    input_dim = X_train.shape[1]
    vae = VAE(input_dim, latent_dim, feature_map)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    # Initialize weights explicitly
    vae(tf.zeros((1, input_dim), dtype=tf.float32))
    
    dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(len(X_train)).batch(batch_size)
    
    history = {"total_loss": [], "recon_loss": [], "kl_loss": [], "beta": []}
    
    for epoch in range(epochs):
        if check_stop_callback and check_stop_callback():
            print(f"Training stopped early by user at epoch {epoch}/{epochs}")
            break
            
        beta = get_beta(epoch, epochs)
        
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        batches = 0
        
        for batch_data in dataset:
            losses = train_step(vae, batch_data, optimizer, feature_map, beta)
            epoch_total += float(losses["total_loss"])
            epoch_recon += float(losses["recon_loss"])
            epoch_kl += float(losses["kl_loss"])
            batches += 1
            
        avg_total = epoch_total / max(1, batches)
        avg_recon = epoch_recon / max(1, batches)
        avg_kl = epoch_kl / max(1, batches)
        
        history["total_loss"].append(avg_total)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)
        history["beta"].append(beta)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_total:.4f} - Recon: {avg_recon:.4f} - KL: {avg_kl:.4f} - Beta: {beta:.3f}")
        
        if progress_callback:
            progress_callback(epoch + 1, epochs, avg_total, avg_recon, avg_kl)
        
    return vae, history

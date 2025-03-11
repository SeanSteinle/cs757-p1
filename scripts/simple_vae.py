#This script is a distilled version of `notebooks/simple_vae.ipynb` where we only keep the model's core components.

#IMPORTS
import librosa, os #audio processing and file system parsing
import librosa.display
import numpy as np #math library
import tensorflow as tf #for model building
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt # for visualization
import pandas as pd #for data analysis / prep
import IPython.display as ipd #for sound output

# Parameters
LATENT_DIM = 64  # Latent space dimension
INPUT_SHAPE = (128, 216, 1)  # (Mel bins, Time steps, Channels)

class ResizeLayer(layers.Layer):
    """"Does a hard resize before decoder output. See the differences between a reshape and a resize here: https://numpy.org/doc/stable/reference/generated/numpy.resize.html."""
    def __init__(self, target_size):
        super(ResizeLayer, self).__init__()
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)

# Reparameterization Trick
def sampling(args):
    """Reparameterization trick: z = mu + exp(log_var / 2) * epsilon"""
    mu, log_var = args
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var * 0.5) * epsilon

# VAE Model Class
class VAE(Model):
    def __init__(self, input_shape=INPUT_SHAPE, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = self.build_encoder(input_shape, latent_dim)

        # Decoder
        self.decoder = self.build_decoder(latent_dim, input_shape)

    def build_encoder(self, input_shape, latent_dim):
        """Builds the VAE Encoder."""
        inputs = layers.Input(shape=input_shape)

        x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)

        mu = layers.Dense(latent_dim, name="latent_mu")(x)
        log_var = layers.Dense(latent_dim, name="latent_log_var")(x)
        z = layers.Lambda(sampling, name="latent_sample")([mu, log_var])

        return Model(inputs, [mu, log_var, z], name="Encoder")

    def build_decoder(self, latent_dim, output_shape):
        decoder_inputs = layers.Input(shape=(latent_dim,))
    
        # Adjust output size based on input time_steps (216)
        x = layers.Dense(16 * 27 * 128, activation="relu")(decoder_inputs)  # 16x27 ensures 216 in time dim
        x = layers.Reshape((16, 27, 128))(x)  # Ensure correct shape
    
        x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
    
        # Ensure output shape is (128, 216, 1)
        outputs = layers.Conv2DTranspose(1, (3, 3), padding="same", activation="sigmoid")(x)
        outputs = ResizeLayer((128, 216))(outputs)  # Resize to match the input shape
    
        return Model(decoder_inputs, outputs, name="Decoder")

    def call(self, inputs):
        """Forward pass through the VAE."""
        mu, log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def compute_loss(self, inputs):
        """Computes the VAE loss function."""
        mu, log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Reconstruction loss
        recon_loss = tf.reduce_mean(tf.keras.losses.mse(inputs, reconstructed))

        # KL Divergence loss
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))

        return recon_loss + kl_loss

    def train_step(self, data):
        """Custom training step."""
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}
    
def generate_samples(model, num_samples=10):
    """Convenience function for generating num_samples from model."""
    random_latent_vectors = tf.random.normal(shape=(num_samples, LATENT_DIM)) # Sample random latent vectors from the prior (e.g., normal distribution)
    generated_samples = model.decoder(random_latent_vectors) # Use the decoder to generate samples from the random latent vectors
    generated_samples = [s.numpy().squeeze() for s in generated_samples] #convert to numpy, reshape
    return generated_samples


from dataprep import load_music_df, create_tf_dataset, load_audio_to_mel

if __name__ == "__main__":
    #load data
    music_df = load_music_df("../data/genres_original/")
    songs_dataset = create_tf_dataset(music_df)

    #create and compile VAE
    vae = VAE()
    vae.compile(optimizer=tf.keras.optimizers.Adam())

    #train and create samples
    vae.fit(songs_dataset, epochs=20)
    generated_samples = generate_samples(vae, num_samples=5)

    #TD: save out model, generated samples
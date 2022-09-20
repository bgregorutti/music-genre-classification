"""
Variational autoencoder for image generation.

Based on https://keras.io/examples/generative/vae/
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Layer, Reshape, Conv2DTranspose, GlobalAveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.metrics import Mean
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy


class Sampling(Layer):
    """
    Uses (z_mean, z_log_var) to sample z, the vector encoding an image.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(Model):
    """
    Main class
    """
    def __init__(self, encoder, decoder, **kwargs):
        """
        Constructor

        Args:
            encoder: the encoder model, object of class tensorflow.keras.models.Model
            decoder: the decoder model, object of class tensorflow.keras.models.Model
        """
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        """
        The metrics used for optimization
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """
        Training method
        """

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(data, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def conv_encoder(input_shape, latent_dim=2):
    """
    Encoder model

    Args:
        input_shape: the shape of the input images
        latent_dim: the latent dimension of the network, default: 2
    
    Returns:
        An object of class tensorflow.keras.models.Model
    """

    # Input layer
    encoder_inputs = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")(encoder_inputs)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Dropout(.3)(x)

    # Flatten / pooling and fully connected layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)

    # Latent representation, mean and log-variance
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    # Sampling on the latent space
    z = Sampling()([z_mean, z_log_var])
    
    return Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def conv_decoder(img_shape, latent_dim=2, filters=[128, 32]):
    """
    Encoder model

    Args:
        latent_dim: the latent dimension of the network, default: 2
    
    Returns:
        An object of class tensorflow.keras.models.Model
    """
    # Final image shape needed
    row, col, channels = img_shape

    # Input layer
    decoder_inputs = Input(shape=(latent_dim,))

    # Fully connected layer and reshape
    target_shape = (
        int(np.ceil(row / 4)),
        int(np.ceil(col / 4)),
        filters[0]
    )
    units = target_shape[0] * target_shape[1] * target_shape[2]
    x = Dense(units, activation="relu")(decoder_inputs)
    x = Reshape(target_shape)(x)

    # Transposed convolutional layers
    x = Conv2DTranspose(filters[0], (3, 3), activation="relu", strides=2, padding="same")(x)
    x = Conv2DTranspose(filters[1], (3, 3), activation="relu", strides=2, padding="same")(x)

    # Network output
    decoder_outputs = Conv2D(channels, (3, 3), activation="sigmoid", padding="same")(x)

    return Model(decoder_inputs, decoder_outputs, name="decoder")

def run():
    input_shape = (28, 28, 1)
    latent_dim = 2
    
    # Get the encoder
    encoder = conv_encoder(input_shape, latent_dim)
    encoder.summary()
    
    # Get the decoder
    decoder = conv_decoder(latent_dim)
    decoder.summary()
    
    # Compile the Autoencoder
    vae = VAE(encoder, decoder)
    vae.compile(optimizer="adam")

    # Load the data and fit the model
    (x_train, y_train), (x_test, _) = tf.keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
    vae.fit(mnist_digits, epochs=10, batch_size=128)

    # Get the latent representation and the bounds
    latent_space = encoder.predict(mnist_digits)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14,12))
    plt.scatter(latent_space[:,0], latent_space[:,1], s=2, c=y_train, cmap='hsv')
    plt.colorbar()
    plt.grid()
    plt.show()

    # Use the decoder to randomly generate a new image
    # new_images = decoder.predict(new_points)

if __name__ == "__main__":
    run()

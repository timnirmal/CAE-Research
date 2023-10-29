import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os


class ConvolutionalAutoencoder:
    """A Convolutional Autoencoder (CAE) implemented in TensorFlow/Keras."""

    def __init__(self, input_shape=(28, 28, 1), encoder_filters=None, decoder_filters=None,
                 kernel_size=(3, 3), pooling_size=(2, 2), upsampling_size=(2, 2),
                 activation='relu', initializer='glorot_uniform'):
        """Initialize the CAE with the given configuration."""
        if decoder_filters is None:
            decoder_filters = [64, 32]
        if encoder_filters is None:
            encoder_filters = [32, 64]
        if len(input_shape) != 3:
            raise ValueError("Input shape should have three dimensions (height, width, channels).")

        self.input_shape = input_shape
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.upsampling_size = upsampling_size
        self.activation = activation
        self.initializer = initializer

        self.autoencoder = self.build_autoencoder()

    def build_encoder(self, x):
        """Build the encoder portion of the autoencoder."""
        for filters in self.encoder_filters:
            x = Conv2D(filters, self.kernel_size, activation=self.activation,
                       padding='same', kernel_initializer=self.initializer)(x)
            x = MaxPooling2D(self.pooling_size, padding='same')(x)
        return x

    def build_decoder(self, x):
        """Build the decoder portion of the autoencoder."""
        for filters in reversed(self.decoder_filters):
            x = Conv2D(filters, self.kernel_size, activation=self.activation,
                       padding='same', kernel_initializer=self.initializer)(x)
            x = UpSampling2D(self.upsampling_size)(x)
        return x

    def build_autoencoder(self):
        """Build the autoencoder using the encoder and decoder."""
        input_img = Input(shape=self.input_shape)
        encoded = self.build_encoder(input_img)
        decoded = self.build_decoder(encoded)
        autoencoder = Model(input_img, decoded)
        return autoencoder

    def compile(self, optimizer='adam', loss='binary_crossentropy'):
        """Compile the autoencoder model."""
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    def fit(self, x_train, x_val, epochs=50, batch_size=256, verbose=1, callbacks=None, plot_metrics=True, log_dir='./logs'):
        """Fit the autoencoder model to the training data."""
        # Set up logging and callbacks
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint = ModelCheckpoint(filepath=os.path.join(log_dir, 'model_best_weights.h5'),
                                     save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(patience=10)

        default_callbacks = [tensorboard, checkpoint, early_stopping]
        if callbacks:
            default_callbacks.extend(callbacks)

        # Train the model
        history = self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                       validation_data=(x_val, x_val), verbose=verbose, callbacks=default_callbacks)

        # Optionally plot training metrics
        if plot_metrics:
            self.plot_metrics(history, log_dir)

        return history

    def plot_metrics(self, history, log_dir, show=True, save=True):
        """Plot training and validation metrics."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot training & validation loss values
        axs[0].plot(history.history['loss'])
        axs[0].plot(history.history['val_loss'])
        axs[0].set_title('Model loss')
        axs[0].set_ylabel('Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(['Train', 'Validation'], loc='upper left')

        # Save or display the plot
        plot_path = os.path.join(log_dir, 'training_metrics.png')
        if save:
            plt.savefig(plot_path)
        if show:
            plt.show()

    def predict(self, x_test):
        """Predict the output of the autoencoder model."""
        return self.autoencoder.predict(x_test)

    def evaluate(self, x_test):
        """Evaluate the autoencoder model."""
        return self.autoencoder.evaluate(x_test, x_test)

    def summary(self):
        """Print the summary of the autoencoder model."""
        self.autoencoder.summary()

    def save_model(self, model_path):
        """Save the autoencoder model."""
        self.autoencoder.save(model_path)

    def save_weights(self, model_path):
        """Save the autoencoder model weights."""
        self.autoencoder.save_weights(model_path)

    def load_model(self, model_path):
        """Load the autoencoder model."""
        self.autoencoder = tf.keras.models.load_model(model_path)

    def load_weights(self, model_path):
        """Load the autoencoder model weights."""
        self.autoencoder.load_weights(model_path)

    def get_encoder(self):
        """Return the encoder model with the currently loaded weights."""
        if self.autoencoder is None:
            raise ValueError("No weights have been loaded. Load weights before extracting the encoder architecture.")
        encoder = Model(self.autoencoder.input, self.autoencoder.layers[len(self.encoder_filters)*2 - 1].output)
        return encoder

    def get_encoder_arch(self, model_path):
        """Return the encoder model with the currently loaded weights."""
        encoder = tf.keras.models.load_model(model_path)
        encoder = Model(encoder.input, encoder.layers[len(self.encoder_filters)*2 - 1].output)
        return encoder


# Usage:
# cae = ConvolutionalAutoencoder()
# cae.compile()
# cae.fit(x_train, x_test)

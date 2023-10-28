import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model

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

    def fit(self, x_train, x_val, epochs=50, batch_size=256, verbose=1, callbacks=None):
        """Fit the autoencoder model to the training data."""
        self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                             validation_data=(x_val, x_val), verbose=verbose, callbacks=callbacks)


# Usage:
# cae = ConvolutionalAutoencoder()
# cae.compile()
# cae.fit(x_train, x_test)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model

class ConvolutionalAutoencoder:
    def __init__(self, input_shape=(28, 28, 1), encoder_filters=[32, 64], decoder_filters=[64, 32],
                 kernel_size=(3, 3), pooling_size=(2, 2), upsampling_size=(2, 2)):
        self.input_shape = input_shape
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.upsampling_size = upsampling_size
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        input_img = Input(shape=self.input_shape)
        x = input_img
        for filters in self.encoder_filters:
            x = Conv2D(filters, self.kernel_size, activation='relu', padding='same')(x)
            x = MaxPooling2D(self.pooling_size, padding='same')(x)

        encoded = x

        for filters in reversed(self.decoder_filters):
            x = Conv2D(filters, self.kernel_size, activation='relu', padding='same')(x)
            x = UpSampling2D(self.upsampling_size)(x)

        decoded = Conv2D(self.input_shape[-1], self.kernel_size, activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        return autoencoder

    def compile(self, optimizer='adam', loss='binary_crossentropy'):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    def fit(self, x_train, x_val, epochs=50, batch_size=256):
        self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_val, x_val))

# Usage:
# cae = ConvolutionalAutoencoder(cae_config)
# cae.compile()
# cae.fit(x_train, x_test)

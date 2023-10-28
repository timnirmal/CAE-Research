import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model

class ConvolutionalAutoencoder:
    def __init__(self, config):
        self.config = config
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        input_img = Input(shape=self.config['input_shape'])
        x = input_img
        for i, filters in enumerate(self.config['encoder_filters']):
            x = Conv2D(filters, self.config['kernel_size'], activation='relu', padding='same')(x)
            x = MaxPooling2D(self.config['pooling_size'], padding='same')(x)

        encoded = x

        for i, filters in enumerate(reversed(self.config['decoder_filters'])):
            x = Conv2D(filters, self.config['kernel_size'], activation='relu', padding='same')(x)
            x = UpSampling2D(self.config['upsampling_size'])(x)

        decoded = Conv2D(self.config['input_shape'][-1], self.config['kernel_size'], activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        return autoencoder

    def compile(self, optimizer='adam', loss='binary_crossentropy'):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

    def fit(self, x_train, x_val, epochs=50, batch_size=256):
        self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_val, x_val))

# Usage:
# cae_config = {...}
# cae = ConvolutionalAutoencoder(cae_config)
# cae.compile()
# cae.fit(x_train, x_test)

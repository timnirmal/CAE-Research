import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input

def create_cae(config):
    # Define the encoder
    input_img = Input(shape=config['input_shape'])
    x = input_img
    for i, filters in enumerate(config['encoder_filters']):
        x = Conv2D(filters, config['kernel_size'], activation='relu', padding='same')(x)
        x = MaxPooling2D(config['pooling_size'], padding='same')(x)

    encoded = x

    # Define the decoder
    for i, filters in enumerate(reversed(config['decoder_filters'])):
        x = Conv2D(filters, config['kernel_size'], activation='relu', padding='same')(x)
        x = UpSampling2D(config['upsampling_size'])(x)

    decoded = Conv2D(config['input_shape'][-1], config['kernel_size'], activation='sigmoid', padding='same')(x)

    # Combine encoder and decoder into one model
    autoencoder = tf.keras.Model(input_img, decoded)
    return autoencoder

# Example configuration dictionary
cae_config = {
    'input_shape': (28, 28, 1),
    'encoder_filters': [32, 64],
    'decoder_filters': [64, 32],
    'kernel_size': (3, 3),
    'pooling_size': (2, 2),
    'upsampling_size': (2, 2)
}

cae_model = create_cae(cae_config)
cae_model.compile(optimizer='adam', loss='binary_crossentropy')
cae_model.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))


from tensorflow.keras.layers import Flatten, Dense

def create_cnn(config):
    input_img = Input(shape=config['input_shape'])
    x = input_img
    for filters in config['conv_filters']:
        x = Conv2D(filters, config['kernel_size'], activation='relu')(x)
        x = MaxPooling2D(config['pooling_size'])(x)

    x = Flatten()(x)
    for units in config['dense_units']:
        x = Dense(units, activation='relu')(x)

    output = Dense(config['num_classes'], activation='softmax')(x)

    cnn_model = tf.keras.Model(input_img, output)
    return cnn_model

# Example configuration dictionary
cnn_config = {
    'input_shape': (28, 28, 1),
    'conv_filters': [32, 64],
    'kernel_size': (3, 3),
    'pooling_size': (2, 2),
    'dense_units': [64],
    'num_classes': 10
}

cnn_model = create_cnn(cnn_config)
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# cnn_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

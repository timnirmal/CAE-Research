from tensorflow.keras.layers import Flatten, Dense

class ConvolutionalNeuralNetwork:
    def __init__(self, config):
        self.config = config
        self.cnn_model = self.build_cnn()

    def build_cnn(self):
        input_img = Input(shape=self.config['input_shape'])
        x = input_img
        for filters in self.config['conv_filters']:
            x = Conv2D(filters, self.config['kernel_size'], activation='relu')(x)
            x = MaxPooling2D(self.config['pooling_size'])(x)

        x = Flatten()(x)
        for units in self.config['dense_units']:
            x = Dense(units, activation='relu')(x)

        output = Dense(self.config['num_classes'], activation='softmax')(x)

        cnn_model = Model(input_img, output)
        return cnn_model

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.cnn_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=64):
        self.cnn_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))

# Usage:
# cnn_config = {...}
# cnn = ConvolutionalNeuralNetwork(cnn_config)
# cnn.compile()
# cnn.fit(x_train, y_train, x_test, y_test)

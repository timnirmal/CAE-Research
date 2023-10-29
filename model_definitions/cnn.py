from keras import Input, Model
from keras.layers import Flatten, Dense
from keras.src.layers import Conv2D, MaxPooling2D


class ConvolutionalNeuralNetwork:
    """A Convolutional Neural Network (CNN) implemented in TensorFlow/Keras."""

    def __init__(self, input_shape=(28, 28, 1), conv_filters=None, kernel_size=(3, 3),
                 pooling_size=(2, 2), dense_units=None, num_classes=10,
                 activation='relu', initializer='glorot_uniform'):
        """Initialize the CNN with the given configuration."""
        if dense_units is None:
            dense_units = [64]
        if conv_filters is None:
            conv_filters = [32, 64]
        if len(input_shape) != 3:
            raise ValueError("Input shape should have three dimensions (height, width, channels).")

        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.dense_units = dense_units
        self.num_classes = num_classes
        self.activation = activation
        self.initializer = initializer

        self.cnn_model = self.build_cnn()

    def build_cnn(self):
        """Build the CNN model."""
        input_img = Input(shape=self.input_shape)
        x = input_img
        for filters in self.conv_filters:
            x = Conv2D(filters, self.kernel_size, activation=self.activation,
                       kernel_initializer=self.initializer)(x)
            x = MaxPooling2D(self.pooling_size)(x)

        x = Flatten()(x)
        for units in self.dense_units:
            x = Dense(units, activation=self.activation,
                      kernel_initializer=self.initializer)(x)

        output = Dense(self.num_classes, activation='softmax')(x)

        cnn_model = Model(input_img, output)
        return cnn_model

    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        """Compile the CNN model."""
        self.cnn_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=64, verbose=1, callbacks=None):
        """Fit the CNN model to the training data."""
        self.cnn_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                           validation_data=(x_val, y_val), verbose=verbose, callbacks=callbacks)

# Usage:
# cnn = ConvolutionalNeuralNetwork(cnn_config)
# cnn.compile()
# cnn.fit(x_train, y_train, x_test, y_test)

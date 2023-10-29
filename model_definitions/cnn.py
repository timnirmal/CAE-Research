import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.models import Model
import os
import mlflow
import mlflow.keras
from model_definitions.base import BaseModel


class ConvolutionalNeuralNetwork(BaseModel):
    def __init__(self, input_shape=(28, 28, 1), conv_filters=None, kernel_size=(3, 3),
                 pooling_size=(2, 2), dense_units=None, num_classes=10,
                 activation='relu', initializer='glorot_uniform'):
        super().__init__(input_shape)
        if dense_units is None:
            dense_units = [64]
        if conv_filters is None:
            conv_filters = [32, 64]

        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.dense_units = dense_units
        self.num_classes = num_classes
        self.activation = activation
        self.initializer = initializer

        self.model = self.build_cnn()

    def build_cnn(self):
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

    def fit(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=64, verbose=1,
            callbacks=None, plot_metrics=True, log_dir='./logs', experiment_name='cnn_experiment'):
        super().fit(x_train, y_train, x_val, y_val, epochs=epochs, batch_size=batch_size,
                    verbose=verbose, callbacks=callbacks, plot_metrics=plot_metrics, log_dir=log_dir,
                    experiment_name=experiment_name)

# Usage:
# cnn = ConvolutionalNeuralNetwork()
# cnn.compile()
# cnn.fit(x_train, y_train, x_val, y_val)  # Training CNN

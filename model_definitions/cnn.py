from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.models import Model
from model_definitions.base import BaseModel


class ConvolutionalNeuralNetwork(BaseModel):
    def __init__(self, input_shape=(28, 28, 1), conv_filters=None, kernel_size=(3, 3),
                 pooling_size=(2, 2), dense_units=None, num_classes=10,
                 activation='relu', initializer='glorot_uniform'):
        """
        Initialize the ConvolutionalNeuralNetwork with the given parameters.
        :param input_shape: Tuple specifying the shape of input data, default is (28, 28, 1).
        :param conv_filters: List of integers specifying the number of filters for each Conv2D layer,
                             default is [32, 64].
        :param kernel_size: Tuple specifying the kernel size for Conv2D layers, default is (3, 3).
        :param pooling_size: Tuple specifying the pool size for MaxPooling2D layers, default is (2, 2).
        :param dense_units: List of integers specifying the number of units for each Dense layer,
                            default is [64].
        :param num_classes: Integer, number of classes for classification, default is 10.
        :param activation: Activation function to use, default is 'relu'.
        :param initializer: Initializer for the kernel weights, default is 'glorot_uniform'.
        """
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
        """
        Build the Convolutional Neural Network.
        :return: CNN model.
        """
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

    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=10, batch_size=64, verbose=1,
            callbacks=None, plot=True, log_dir='./logs', experiment_name='cnn_experiment'):
        """
        Fit the CNN model to the training data and validate on the validation data if provided.
        :param x_train: Training data features.
        :param y_train: Training data labels.
        :param x_val: Validation data features, default is None.
        :param y_val: Validation data labels, default is None.
        :param epochs: Number of epochs to train for, default is 10.
        :param batch_size: Batch size for training, default is 64.
        :param verbose: Verbosity mode, 0 = silent, 1 = progress bar, 2 = one line per epoch. Default is 1.
        :param callbacks: List of callbacks to apply during training, default is None.
        :param plot: Boolean, whether to plot metrics after training, default is True.
        :param log_dir: Directory for saving logs and model weights, default is './logs'.
        :param experiment_name: Name of the MLFlow experiment, default is 'cnn_experiment'.
        :return: History object containing training/validation loss and metric values.
        """
        return super().fit(x_train, y_train, x_val, y_val, epochs, batch_size, verbose,
                           callbacks, plot, log_dir, experiment_name)

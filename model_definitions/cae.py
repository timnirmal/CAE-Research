import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.models import Model
from model_definitions.base import BaseModel


class ConvolutionalAutoencoder(BaseModel):
    def __init__(self, input_shape=(28, 28, 1), encoder_filters=None, decoder_filters=None,
                 kernel_size=(3, 3), pooling_size=(2, 2), upsampling_size=(2, 2),
                 activation='relu', initializer='glorot_uniform'):
        """
        Initialize the ConvolutionalAutoencoder with the given parameters.
        :param input_shape: Tuple specifying the shape of input data, default is (28, 28, 1).
        :param encoder_filters: List of integers specifying the number of filters for each Conv2D layer in the encoder,
                                default is [32, 64].
        :param decoder_filters: List of integers specifying the number of filters for each Conv2D layer in the decoder,
                                default is [64, 32].
        :param kernel_size: Tuple specifying the kernel size for Conv2D layers, default is (3, 3).
        :param pooling_size: Tuple specifying the pool size for MaxPooling2D layers, default is (2, 2).
        :param upsampling_size: Tuple specifying the size for UpSampling2D layers, default is (2, 2).
        :param activation: Activation function to use, default is 'relu'.
        :param initializer: Initializer for the kernel weights, default is 'glorot_uniform'.
        """
        super().__init__(input_shape)
        if decoder_filters is None:
            decoder_filters = [64, 32]
        if encoder_filters is None:
            encoder_filters = [32, 64]
        if len(input_shape) != 3:
            raise ValueError("Input shape should have three dimensions (height, width, channels).")

        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.upsampling_size = upsampling_size
        self.activation = activation
        self.initializer = initializer

        self.model = self.build_autoencoder()

    def build_encoder(self, x):
        """
        Build the encoder portion of the autoencoder.
        :param x: Input tensor.
        :return: Tensor representing the encoded features.
        """
        for filters in self.encoder_filters:
            x = Conv2D(filters, self.kernel_size, activation=self.activation,
                       padding='same', kernel_initializer=self.initializer)(x)
            x = MaxPooling2D(self.pooling_size, padding='same')(x)
        return x

    def build_decoder(self, x):
        """
        Build the decoder portion of the autoencoder.
        :param x: Encoded input tensor.
        :return: Tensor representing the decoded output.
        """
        for filters in reversed(self.decoder_filters):
            x = Conv2D(filters, self.kernel_size, activation=self.activation,
                       padding='same', kernel_initializer=self.initializer)(x)
            x = UpSampling2D(self.upsampling_size)(x)
        return x

    def build_autoencoder(self):
        """
        Build the autoencoder using the encoder and decoder.
        :return: Autoencoder model.
        """
        input_img = Input(shape=self.input_shape)
        encoded = self.build_encoder(input_img)
        decoded = self.build_decoder(encoded)
        autoencoder = Model(input_img, decoded)
        return autoencoder

    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=50, batch_size=256, verbose=1,
            callbacks=None, plot=True, log_dir='./logs', experiment_name='experiment'):
        """
        Fit the autoencoder model to the training data.
        :param x_train: Training data.
        :param x_val: Validation data.
        :param epochs: Number of epochs to train for, default is 50.
        :param batch_size: Batch size for training, default is 256.
        :param verbose: Verbosity mode, 0 = silent, 1 = progress bar, 2 = one line per epoch. Default is 1.
        :param callbacks: List of callbacks to apply during training, default is None.
        :param plot: Boolean, whether to plot metrics after training, default is True.
        :param log_dir: Directory for saving logs and model weights, default is './logs'.
        :return: History object containing training/validation loss and metric values.
        """
        return super().fit(x_train, x_train, x_val if x_val is not None else x_train,
                           y_val if y_val is not None else x_val, epochs, batch_size, verbose,
                           callbacks, plot, log_dir, experiment_name)

    def get_encoder(self):
        """Return the encoder model with the currently loaded weights."""
        if self.model is None:
            raise ValueError("No weights have been loaded. Load weights before extracting the encoder architecture.")
        encoder = Model(self.model.input, self.model.layers[len(self.encoder_filters) * 2 - 1].output)
        return encoder

    def get_encoder_by_path(self, model_path):
        """
        Create an instance of the ConvolutionalAutoencoder and load the saved weights.
        Then, return the encoder part of the model.
        :param model_path: Path to the saved weights.
        :return: Encoder model.
        """
        # First, create an instance of ConvolutionalAutoencoder with the same architecture
        temp_cae = ConvolutionalAutoencoder(input_shape=self.input_shape)
        # Load the saved weights into this model
        temp_cae.load_weights(model_path)

        # Extract the encoder part from this model
        encoder = Model(temp_cae.model.input, temp_cae.model.layers[len(temp_cae.encoder_filters) * 2 - 1].output)
        return encoder

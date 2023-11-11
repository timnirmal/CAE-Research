import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


def plot_metrics(history, log_dir, save=True, show=False):
    """
    Plot training and validation metrics.
    :param history: History object containing training/validation loss and metric values.
    :param log_dir: Directory for saving the plot.
    :param save: Boolean, whether to save the plot, default is True.
    :param show: Boolean, whether to show the plot, default is False.
    :return: Path where the plot is saved.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('Model loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    plot_path = os.path.join(log_dir, 'training_metrics.png')
    if save:
        plot_path = os.path.join(log_dir, 'training_metrics.png')
        plt.savefig(plot_path)
    if show:
        plt.show()
    return plot_path


class BaseModel:
    def __init__(self, input_shape):
        """
        Initialize the BaseModel with the given input shape.
        :param input_shape: Tuple specifying the shape of input data.
        """
        self.input_shape = input_shape
        self.model = None  # to be defined by subclasses

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=None):
        """
        Compile the model with the given optimizer and loss function.
        :param optimizer: String or optimizer instance, default is 'adam'.
        :param loss: String or loss instance, default is 'binary_crossentropy'.
        """
        if metrics is None:
            self.model.compile(optimizer=optimizer, loss=loss)
        else:
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=50, batch_size=256, verbose=1,
            callbacks=None, plot=True, log_dir='./logs', experiment_name='experiment'):
        """
        Fit the model to the training data and validate on the validation data if provided.
        :param x_train: Training data features.
        :param y_train: Training data labels.
        :param x_val: Validation data features, default is None.
        :param y_val: Validation data labels, default is None.
        :param epochs: Number of epochs to train for, default is 50.
        :param batch_size: Batch size for training, default is 256.
        :param verbose: Verbosity mode, 0 = silent, 1 = progress bar, 2 = one line per epoch. Default is 1.
        :param callbacks: List of callbacks to apply during training, default is None.
        :param plot: Boolean, whether to plot metrics after training, default is True.
        :param log_dir: Directory for saving logs and model weights, default is './logs'.
        :param experiment_name: Name of the MLFlow experiment, default is 'experiment'.
        :return: History object containing training/validation loss and metric values.
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint = ModelCheckpoint(filepath=os.path.join(log_dir, 'model_best_weights.h5'),
                                     save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(patience=10)

        default_callbacks = [tensorboard, checkpoint, early_stopping]
        if callbacks:
            default_callbacks.extend(callbacks)

        validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                     validation_data=validation_data, verbose=verbose, callbacks=default_callbacks)

            for epoch, metrics in enumerate(zip(history.history['loss'], history.history.get('val_loss', []))):
                train_loss, val_loss = metrics
                mlflow.log_metric('train_loss', train_loss, step=epoch)
                if val_loss:
                    mlflow.log_metric('val_loss', val_loss, step=epoch)

            if plot:
                plot_path = plot_metrics(history, log_dir)
                mlflow.log_artifact(plot_path)

            mlflow.keras.log_model(self.model, "models")

        return history

    def predict(self, x_test):
        """
        Make predictions on the given test data.
        :param x_test: Test data features.
        :return: Predictions.
        """
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        """
        Evaluate the model on the given test data.
        :param x_test: Test data features.
        :param y_test: Test data labels.
        :return: Evaluation metrics.
        """
        return self.model.evaluate(x_test, y_test)

    def summary(self):
        """
        Print the summary of the model.
        """
        self.model.summary()

    def save_model(self, model_path):
        """
        Save the model to the given path.
        :param model_path: Path for saving the model.
        """
        self.model.save(model_path)

    def save_weights(self, model_path):
        """
        Save the model weights to the given path.
        :param model_path: Path for saving the model weights.
        """
        self.model.save_weights(model_path)

    def load_model(self, model_path):
        """
        Load the model from the given path.
        :param model_path: Path for loading the model.
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model file {model_path} does not exist.")
        self.model = tf.keras.models.load_model(model_path)

    def load_weights(self, model_path):
        """
        Load the model weights from the given path.
        :param model_path: Path for loading the model weights.
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model weights file {model_path} does not exist.")
        self.model.load_weights(model_path)

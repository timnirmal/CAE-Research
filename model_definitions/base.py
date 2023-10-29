import mlflow
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os


class BaseModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None  # to be defined by subclasses

    def compile(self, optimizer='adam', loss='binary_crossentropy'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=50, batch_size=256, verbose=1,
            callbacks=None, plot_metrics=True, log_dir='./logs', experiment_name='experiment'):
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

        # Start MLFlow experiment
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                                     validation_data=validation_data, verbose=verbose, callbacks=default_callbacks)

            # Log metrics to MLFlow
            for epoch, metrics in enumerate(zip(history.history['loss'], history.history.get('val_loss', []))):
                train_loss, val_loss = metrics
                mlflow.log_metric('train_loss', train_loss, step=epoch)
                if val_loss:
                    mlflow.log_metric('val_loss', val_loss, step=epoch)

            # Optionally save and plot training metrics
            if plot_metrics:
                plot_path = self.plot_metrics(history, log_dir)
                mlflow.log_artifact(plot_path)

            # Log model to MLFlow
            mlflow.keras.log_model(self.model, "models")

        return history

    def plot_metrics(self, history, log_dir, save=True, show=False):
        """Plot training and validation metrics."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot training & validation loss values
        axs[0].plot(history.history['loss'])
        axs[0].plot(history.history['val_loss'])
        axs[0].set_title('Model loss')
        axs[0].set_ylabel('Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(['Train', 'Validation'], loc='upper left')

        # Save or display the plot
        plot_path = os.path.join(log_dir, 'training_metrics.png')
        if save:
            plot_path = os.path.join(log_dir, 'training_metrics.png')
            plt.savefig(plot_path)
        if show:
            plt.show()
        return plot_path

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def summary(self):
        self.model.summary()

    def save_model(self, model_path):
        self.model.save(model_path)

    def save_weights(self, model_path):
        self.model.save_weights(model_path)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def load_weights(self, model_path):
        self.model.load_weights(model_path)

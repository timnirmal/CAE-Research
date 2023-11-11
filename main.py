import tensorflow as tf

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0  # Normalize pixel values to [0, 1]

# Reshape the data to include the channel dimension
x_train = x_train[..., tf.newaxis]
x_val = x_val[..., tf.newaxis]

# train with only 1000 samples
x_train = x_train[:1000]
y_train = y_train[:1000]
x_val = x_val[:1000]
y_val = y_val[:1000]

from model_definitions.cae import ConvolutionalAutoencoder

cae = ConvolutionalAutoencoder(input_shape=(28, 28, 1))
cae.compile(optimizer='adam', loss='binary_crossentropy')
cae_history = cae.fit(x_train, None, x_val, None, epochs=10, batch_size=256, verbose=1)
cae.summary()

cae.save_model('cae_model.keras')
cae.save_weights('cae_weights.keras')


from model_definitions.cnn import ConvolutionalNeuralNetwork

cnn = ConvolutionalNeuralNetwork(input_shape=(28, 28, 1), num_classes=10)
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_history = cnn.fit(x_train, y_train, x_val, y_val, epochs=10, batch_size=64, verbose=1)
cnn.summary()

# Evaluate the model
cnn_evaluation = cnn.evaluate(x_val, y_val)

# Make predictions
predictions = cnn.predict(x_val)

cnn.save_model('cnn_model.keras')
cnn.save_weights('cnn_weights.keras')
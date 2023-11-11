import tensorflow as tf
from keras import Model

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

# cae = ConvolutionalAutoencoder(input_shape=(28, 28, 1))
# cae.compile(optimizer='adam', loss='binary_crossentropy')
# cae_history = cae.fit(x_train, None, x_val, None, epochs=10, batch_size=256, verbose=1)
# cae.summary()
#
# cae.save_model('cae_model.keras')
# cae.save_weights('cae_weights.keras')
#
# del cae

cae = ConvolutionalAutoencoder(input_shape=(28, 28, 1))
cae.load_weights('cae_weights.keras')
encoder = cae.get_encoder_by_path('cae_weights.keras')

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

# combine the encoder and classifier
encoder_output = encoder.layers[-1].output

# Assuming the cnn model's first layer is the Conv2D layer
classifier_input = cnn.model.layers[0].input
classifier_output = cnn.model(classifier_input)

combined_output = classifier_output(encoder_output)
combined_model = Model(encoder.input, combined_output)

# 9. save the model
combined_model.save('combined_model.keras')

# 10. evaluate new model
combined_model.evaluate(x_val, y_val)
from keras import Model
from keras.src.layers import Conv2D, Reshape, UpSampling2D
from model_definitions.cae import ConvolutionalAutoencoder
from model_definitions.cnn import ConvolutionalNeuralNetwork

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



# load cae model
cae = ConvolutionalAutoencoder()
cae.load_model('cae_model.keras')
encoder = cae.get_encoder()

# load cnn model
cnn = ConvolutionalNeuralNetwork()
cnn.load_model('cnn_model.keras')

# evaluate the CNN model
print(cnn.model.manual_evaluate(x_val, y_val))

transitional_layer = Conv2D(1, (3, 3), activation='relu', padding='same')(encoder.output)
transitional_layer = UpSampling2D((2, 2))(transitional_layer)

# Now connect the transitional layer's output to the CNN
cnn_output = cnn.model(transitional_layer)

# Create the final combined model
combined_model = Model(inputs=encoder.input, outputs=cnn_output)
# compile the combined model
combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # evaluate the combined model
# combined_model.manual_evaluate(x_val, y_val)




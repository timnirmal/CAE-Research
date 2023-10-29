# # load mnist data
# from keras.datasets import mnist
# from model_definitions.cae import ConvolutionalAutoencoder
# from model_definitions.cnn import ConvolutionalNeuralNetwork
#
# # (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # # reshape and normalize input data
# # x_train = x_train.reshape(-1, 28, 28, 1) / 255.
# # x_test = x_test.reshape(-1, 28, 28, 1) / 255.
# #
# # cae = ConvolutionalAutoencoder()
# # cae.compile()
# # cae.summary()
# # cae.fit(x_train, x_test)
# # cae.save_model('cae.h5')
# # cae.save_weights('cae_weights.h5')
# # # del cae
# # del cae
# cae = ConvolutionalAutoencoder()
# cae.load_weights('cae_weights.h5')
# encoder = cae.get_encoder_arch('cae_weights.h5')
# encoder.summary()
# exit()
#
# # cnn = ConvolutionalNeuralNetwork()
# # cnn.compile()
# # cnn.fit(x_train, y_train, x_test, y_test)
#
# # # test with sample image
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# # sample_image = x_test[0]
# # sample_image = np.expand_dims(sample_image, axis=0)
# # sample_image = np.expand_dims(sample_image, axis=3)
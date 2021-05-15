import os
import imageio
import numpy as np
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, Activation, Concatenate
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import matplotlib.pyplot as plt

class InstanceNormalization(tf.keras.layers.Layer):
    # Initialization of Objects
    def __init__(self, epsilon=1e-5):
        # calling parent's init
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        # Compute Mean and Variance, Axes=[1,2] ensures Instance Normalization
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class DataUtils:
    def __init__(self, sourcePath, resize):
        '''

        :param sourcePath: File path to data source
        '''
        self.data = None
        self.sourcePath = sourcePath
        self.resize = resize
        self.imageSize = None

        self.prepareData()

    def prepareData(self):
        self.readFiles()
        self.dataPreprocess()

    def readFiles(self):
        '''
        function to read the data files from the path specified
        :return: the list of data elements in the form of numpy array
        '''

        # check if the path is valid or not
        if os.path.isdir(self.sourcePath):
            self.data = [imageio.imread(os.path.join(self.sourcePath, f)) for f in os.listdir(self.sourcePath)
                         if f.endswith(('.jpeg', '.jpg', '.png'))]

            self.imageSize = self.data[-1].shape

            if len(self.imageSize) != len(self.resize):
                raise Exception("Size mismatch!!")

        else:
            raise Exception("Path Invalid")

    def dataPreprocess(self):
        # normalize the data between -1 to +1
        normalized_data = (np.asarray(self.data, dtype=np.float32) / 127.5) - 1

        if len(self.imageSize) == 2:
            self.resize = (self.resize[0], self.resize[1], 1)
            self.imageSize = (self.imageSize[0], self.imageSize[1], 1)
            normalized_data.reshape((normalized_data.shape[0],self.imageSize[0],self.imageSize[1], self.imageSize[2]))


        final_shape = (normalized_data.shape[0], self.resize[0], self.resize[1], self.resize[2])

        self.data = np.zeros(final_shape, dtype=np.float32)
        for index, img in enumerate(normalized_data):
            self.data[index, :, :,:] = resize(img, self.resize)


    def get_data(self, batch_size):
        # batch and shuffle the data
        return tf.data.Dataset.from_tensor_slices(self.data).shuffle(self.data.shape[0], seed=42).batch(batch_size)



def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization()(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization()(g)
    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g

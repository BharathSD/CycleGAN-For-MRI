from Utils import DataUtils, resnet_block, InstanceNormalization
from CycleGAN import CycleGAN
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Activation, Input, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
import os


class CycleGAN_TR1_TR2(CycleGAN):

    def __init__(self, input_shape, checkpoint_path):
        self.n_resnet = 3
        super().__init__(input_shape, checkpoint_path)


    def build_generator(self, image_shape):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # image input
        in_image = Input(shape=image_shape)
        # c7s1-64
        g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
        g = InstanceNormalization()(g)
        g = Activation('relu')(g)
        # d128
        g = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization()(g)
        g = Activation('relu')(g)
        # d256
        g = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization()(g)
        g = Activation('relu')(g)
        # R256
        for _ in range(self.n_resnet):
            g = resnet_block(256, g)
        # u128
        g = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization()(g)
        g = Activation('relu')(g)
        # u64
        g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization()(g)
        g = Activation('relu')(g)
        # c7s1-3
        g = Conv2D(1, (7, 7), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization()(g)
        out_image = Activation('tanh')(g)
        # define model
        model = Model(in_image, out_image)
        return model

    def build_discriminator(self, image_shape):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        in_image = Input(shape=image_shape)
        # C64
        d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
        d = InstanceNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
        # define model
        model = Model(in_image, patch_out)
        return model

if __name__ == '__main__':
    path2TR1 = os.path.join(os.getcwd(), 'Tr1', 'TrainT1')
    path2TR2 = os.path.join(os.getcwd(), 'Tr2', 'TrainT2')
    checkpoint_path = os.path.join(os.getcwd(), 'Trained_Model4')

    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    batch_size = 4
    epochs = 100
    # load images from
    images_x = DataUtils(path2TR1, (220,184)).get_data(batch_size)
    images_y = DataUtils(path2TR2, (220,184)).get_data(batch_size)

    sample_x_data = next(iter(images_x))
    sample_y_data = next(iter(images_y))

    cycleGan = CycleGAN_TR1_TR2((220,184,1), checkpoint_path)

    cycleGan.train(images_x, images_y, epochs,plot_results=True, sample_data=(sample_x_data, sample_y_data))

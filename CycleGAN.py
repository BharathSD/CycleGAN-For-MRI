import tensorflow as tf
import matplotlib.pyplot as plt
import time


class CycleGAN:

    def __init__(self, input_shape, checkpoint_path):

        self.discriminator_y = self.build_discriminator(input_shape)
        self.discriminator_x = self.build_discriminator(input_shape)
        self.generator_f = self.build_generator(input_shape)
        self.generator_g = self.build_generator(input_shape)

        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.checkpoint_path = checkpoint_path

        self.ckpt = tf.train.Checkpoint(generator_g=self.generator_g,
                                        generator_f=self.generator_f,
                                        discriminator_x=self.discriminator_x,
                                        discriminator_y=self.discriminator_y,
                                        generator_g_optimizer=self.generator_g_optimizer,
                                        generator_f_optimizer=self.generator_f_optimizer,
                                        discriminator_x_optimizer=self.discriminator_x_optimizer,
                                        discriminator_y_optimizer=self.discriminator_y_optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=3)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5  # mean of losses

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    @staticmethod
    def calc_cycle_loss(real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return 10.0 * loss1

    @staticmethod
    def identity_loss(real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return 0.5 * loss

    @tf.function
    def train_step(self, real_x, real_y):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            # Total generator loss = BCE loss + cycle loss + identity loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            # Discriminator's loss
            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, self.generator_g.trainable_variables))
        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, self.generator_f.trainable_variables))

        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                           self.discriminator_x.trainable_variables))
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                           self.discriminator_y.trainable_variables))

    def build_generator(self, image_shape):
        raise Exception("Not implemented")

    def build_discriminator(self, image_shape):
        raise Exception("Not implemented")

    def train(self, data_x, data_y, epochs: int, save_ckpt: bool = True, plot_results: bool = False, save_results:bool=False,
              sample_data: tuple = None):
        for epoch in range(1, epochs + 1):
            for image_x, image_y in tf.data.Dataset.zip((data_x, data_y)):
                self.train_step(image_x, image_y)

            if save_ckpt:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch', epoch, 'at', ckpt_save_path)

            if plot_results:
                if sample_data is None:
                    raise Exception("sample data is None, Cannot plot results")
                if len(sample_data) != 2:
                    raise Exception("sample data is expected of the form (image_x, image_y) with proper dimensions as "
                                    "trained")
                test_image_x, test_image_y = sample_data[0], sample_data[1]
                predictionX, predictionY = self.generate_images(test_image_x, test_image_y)
                plt.figure(figsize=(8, 4))
                display_list = [test_image_x[0], predictionX[0], test_image_y[0], predictionY[0]]
                title = ['Input Image', 'Predicted Image', 'Input Image', 'Predicted Image']
                for i in range(4):
                    plt.subplot(1, 4, i + 1)
                    plt.title(title[i])
                    plt.imshow(display_list[i].numpy()[:, :, 0], cmap='gray')
                    plt.axis('off')

                plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
                plt.show(block=False)
                time.sleep(5)
                plt.close('all')

    def generate_images(self, image_x, image_y):
        prediction1 = self.generator_g(image_x)
        prediction2 = self.generator_f(image_y)
        return prediction1, prediction2

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# return a list of paths of images
def get_image_path(img_dir):
    image_file_list = os.listdir(img_dir)
    image_path = [os.path.join(img_dir, img_name) for img_name in image_file_list]

    return image_path


# process images
def map_image(image_dir):
    image_raw = tf.io.read_file(image_dir)
    image = tf.image.decode_jpeg(image_raw)

    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(image, (64, 64))
    image = image / 255.0
    image = tf.reshape(image, shape=(64, 64, 3))

    return image


# display a row of images
def display_one_row(disp_images, offset, shape=None):
    for idx, img in enumerate(disp_images):
        plt.subplot(3, 10, offset + idx + 1)
        plt.xticks([])
        plt.yticks([])
        img = np.reshape(img, shape)
        plt.imshow(img)


def display_images(disp_input, disp_predicted):
    plt.figure(figsize=(15, 5))
    display_one_row(disp_input, 0, shape=(64, 64, 3))
    display_one_row(disp_predicted, 10, shape=(64, 64, 3))


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        """
        param inputs: output tensor from the encoder

        return: tensors combined with a random sample
        """
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

        # generate a random tensor
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        combined_tensor = mu + tf.exp(0.5 * sigma) * epsilon

        return combined_tensor


def get_KL_loss(inputs, outputs, mu, sigma):   # inputs, outputs ???
    """
    compute the Kullback-Leibler Divergence
    :param inputs: input of dataset
    :param outputs: output of encoder
    :param mu: mean
    :param sigma: standard deviation
    :return: kl_loss
    """
    kl_loss = 1 + sigma - tf.square(mu) - tf.math.exp(sigma)
    kl_loss = tf.reduce_mean(kl_loss) * (-0.5)

    return kl_loss


def generate_and_sive_images(model, epoch, step, test_input):
    """
    helper function to plot 16 images generated by decoder
    :param model: the decoder which has been trained well
    :param epoch: number of epoch
    :param step: number of step
    :param test_input: random tensor with shape (16, LATENT_DIM)
    """
    predictions = model.predict(test_input)
    fig = plt.figure(figsize=(6,6))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        img = predictions[i, :, :, :] * 255
        img = img.astype('int32')
        plt.imshow(img)
        plt.axis('off')

    fig.suptitle('epoch: {}, step: {}'.format(epoch, step))
    plt.savefig('image_at_epoch_{:04d}_step{:04d}.png'.format(epoch, step))
    plt.show()

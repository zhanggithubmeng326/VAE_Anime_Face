import tensorflow as tf
import numpy as np
import random
import utils
import networks
from IPython import display


def train(epochs, latent_dim, train_ds, vae_model):
    """
    training loop
    1) feed a training batch to the VAE model
    2) compute reconstruction loss (mes_loss)
    3) add kl_loss to the total loss
    4) get the gradients
    5) apply the optimizer to update the weights
    """
    random_vector_for_generation = tf.random.normal(shape=[16, latent_dim])

    # define loss, optimizer and loss metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    loss_metric = tf.keras.metrics.Mean()
    mse_loss = tf.keras.losses.MeanSquaredError()

    for epoch in range(epochs):
        print('Start of epoch {:%d}'.format(epoch))

        for step, x_batch_train in enumerate(train_ds):
            with tf.GradientTape() as tape:
                reconstructed_img = vae_model(x_batch_train)

                # compute reconstruction loss
                flattened_inputs = tf.reshape(x_batch_train, shape=[-1])
                flattened_outputs = tf.reshape(reconstructed_img, shape=[-1])
                loss = mse_loss(flattened_inputs, flattened_outputs) * 64 * 64 * 3
                loss += sum(vae.losses)      # add kl_loss

            gradients = tape.gradient(loss, vae_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, vae_model.trainable_variables))
            loss_metric(loss)

            # display outputs every 100 steps
            if (step+1) % 100 == 0:
                display.clear_output(wait=False)
                utils.generate_and_sive_images(decoder, epoch, step, random_vector_for_generation)
                print('Epoch: %s step: %s, mean_loss: %s' % (epoch, step, loss_metric.result()))

        loss_metric.reset_states()


if __name__ == '__main__':
    np.random.seed(51)

    EPOCHS = 100
    BATCH_SIZE = 2000
    LATENT_DIM = 512
    IMAGE_SIZE = 64
    PATH = 'home/mzhang/...'

    image_path = utils.get_image_path(PATH)
    random.shuffle(image_path)

    # split paths list into train(80%) and validation(20%) sets
    len_path = len(PATH)
    len_train_path = int(len_path * 0.8)

    train_path = image_path[0: len_train_path]
    val_path = image_path[len_train_path:]

    # load the images for train into tensors, create batches and shuffle
    train_dataset = tf.data.Dataset.from_tensor_slices(train_path)
    train_dataset = train_dataset.map(utils.map_image)
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)

    # load the images for validation into tensors, create batches and shuffle
    validation_dataset = tf.data.Dataset.from_tensor_slices(val_path)
    validation_dataset = validation_dataset.map(utils.map_image)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    print(f'number of batches in the train set : {len(train_dataset)}')
    print(f'number of batches in the validation set : {len(validation_dataset)}')

    TRAIN_DS = train_dataset

    encoder, decoder, vae = networks.get_models(input_shape=(64, 64, 3), latent_dim=LATENT_DIM)
    train(epochs=EPOCHS, latent_dim=LATENT_DIM, train_ds=TRAIN_DS, vae_model=vae)
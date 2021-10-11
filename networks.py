import tensorflow as tf
import utils


# define encoder layers
def encoder_layers(inputs, latent_dim):
    """
    define encoder layers
    :param inputs: input images from dataset
    :param latent_dim: dimensionality of the latent variable
    :return: mu -- learned mean vector
             sigma -- learned standard deviation
             features -- shape of the features before flattening
    """
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', name='enconder_conv1')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name='enconder_conv2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu', name='enconder_conv3')(x)
    features = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten(name='encoder_flatten')(features)
    x = tf.keras.layers.Dense(1024, activation='relu', name='encoder_dense')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # dense layers for mu and sigma
    mu = tf.keras.layers.Dense(latent_dim, name='latent_mu')(x)
    sigma = tf.keras.layers.Dense(latent_dim, name='latent_sigma')(x)

    return mu, sigma, features


def encoder_model(input_shape, latent_dim):
    inputs = tf.keras.layers.Input(shape=input_shape)
    mu, sigma, feature = encoder_layers(inputs, latent_dim)
    z = utils.Sampling((mu, sigma))
    model = tf.keras.Model(inputs, outputs=[mu, sigma, z])
    model.summary()

    return model, feature


def decoder_layers(inputs, feature):
    """
    define the decoder layers
    :param inputs: output of encoder
    :param feature: shape of feature before flattening
    :return: tensor containing the decoded output
    """
    units = feature[1] * feature[2] * feature[3]
    x = tf.keras.layers.Dense(units, activation='relu', name='decoder_dense1')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    # reshape output using the dimension of feature shape
    x = tf.keras.layers.Reshape((feature[1], feature[2], feature[3]), name='decoder_reshape')(x)

    # upsample the feature back to the original dimensions
    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu', name='decoder_conv2d_2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name='decoder_conv2d_3')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', name='decoder_conv2d_4')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', activation='sigmoid', name='decoder_final')

    return x


def decoder_model(latent_dim, feature_shape):
    inputs = tf.keras.layers.Input(shape=(latent_dim,))
    outputs = decoder_model(latent_dim, feature_shape)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    return model


def vae_model(encoder, decoder, input_shape):
    """
    define the entire VAE model
    :param encoder: the encoder model
    :param decoder: the decoder model
    :param input_shape: shape of the dataset
    :return:
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    mu, sigma, z = encoder(inputs)
    reconstructed_img = decoder(z)
    model = tf.keras.Model(inputs=inputs, outputs=reconstructed_img)

    kl_loss = utils.get_KL_loss(inputs, z, mu, sigma)
    model.add_loss(kl_loss)

    return model


def get_models(input_shape, latent_dim):
    """ return the encoder, decoder, and vae models """
    encoder, feature = encoder_model(input_shape=input_shape, latent_dim=latent_dim)
    decoder = decoder_model(latent_dim=latent_dim, feature_shape=feature)
    vae = vae_model(encoder=encoder, decoder=decoder, input_shape=input_shape)

    return encoder, decoder, vae







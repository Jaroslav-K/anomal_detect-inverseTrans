import tensorflow as tf
from discriminator import get_discriminator_model
from generator import get_generator_model
from tensorflow.keras import layers


class GAN(tf.keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
    ):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # if isinstance(data, tuple):
        #   real_images = data[0]
        #   target = data[1]
        real_images, target = data

        # sample random noise for generator
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # create fake images
        generated_images = self.generator([random_latent_vectors, target])

        # combine fake and real images
        combined_images = tf.concat([generated_images, real_images], axis=0)
        targets = tf.concat([target, target], axis=0)

        # Assemble labels for both fake and real images
        labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - (important trick but experimental for now)
        # labels += 0.05 * tf.random.uniform(tf.shape(labels)) #noise

        # Train the discriminator

        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_images, targets])
            d_loss = self.loss_fn(labels, predictions)
            # d_loss *= 2
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

        # sample again
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # assemble labels
        misleading_labels = tf.ones((batch_size, 1))
        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(
                [self.generator([random_latent_vectors, target]), target]
            )
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}


def get_gan_model(compile=True):
    latent_dim = 100

    disc = get_discriminator_model()
    gen = get_generator_model()
    gan = GAN(discriminator=disc, generator=gen, latent_dim=latent_dim)

    if compile:
        gan.compile(
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.99),
            loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        )
    return gan

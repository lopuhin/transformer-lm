"""
Training loop, based on
https://www.tensorflow.org/alpha/tutorials/distribute/training_loops
"""
import os

import numpy as np
import tensorflow as tf
import tqdm


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    print('model', model)
    return model


def main():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (
    test_images, test_labels) = fashion_mnist.load_data()

    # Adding a dimension to the array -> new shape == (28, 28, 1)
    # We are doing this because the first layer in our model is a convolutional
    # layer and it requires a 4D input (batch_size, height, width, channels).
    # batch_size dimension will be added later on.
    train_images = train_images[..., None]
    test_images = test_images[..., None]

    # Getting the images in [0, 1] range.
    train_images = train_images / np.float32(255)
    test_images = test_images / np.float32(255)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    BUFFER_SIZE = len(train_images)

    BATCH_SIZE_PER_REPLICA = 4  # 64
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    EPOCHS = 1 # 10
    train_steps_per_epoch = len(train_images) // BATCH_SIZE
    test_steps_per_epoch = len(test_images) // BATCH_SIZE

    with strategy.scope():
        train_iterator = strategy.experimental_make_numpy_iterator(
            (train_images, train_labels), BATCH_SIZE, shuffle=BUFFER_SIZE)

        test_iterator = strategy.experimental_make_numpy_iterator(
            (test_images, test_labels), BATCH_SIZE, shuffle=None)

    # Create a checkpoint directory to store the checkpoints.
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    with strategy.scope():
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        test_loss = tf.keras.metrics.Mean(name='test_loss')

        model = create_model()
        optimizer = tf.keras.optimizers.Adam()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        def train_step(inputs):
            images, labels = inputs

            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)

        def test_step(inputs):
            images, labels = inputs
            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)
            test_loss(t_loss)

        @tf.function
        def distributed_train():
            return strategy.experimental_run(train_step, train_iterator)

        @tf.function
        def distributed_test():
            return strategy.experimental_run(test_step, test_iterator)

        for epoch in range(EPOCHS):

            # TRAIN LOOP
            # Initialize the iterator
            train_iterator.initialize()
            for _ in tqdm.trange(train_steps_per_epoch, desc='train'):
                distributed_train()

            # TEST LOOP
            test_iterator.initialize()
            for _ in tqdm.trange(test_steps_per_epoch, desc='validate'):
                distributed_test()

            if epoch % 2 == 0:
                checkpoint.save(checkpoint_prefix)

            template = 'Epoch {}, Loss: {} Test Loss: {}'
            print(template.format(epoch + 1, train_loss.result(),
                                  test_loss.result()))

            train_loss.reset_states()
            test_loss.reset_states()


if __name__ == '__main__':
    main()
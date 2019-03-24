"""
Training loop, based on
https://www.tensorflow.org/alpha/tutorials/distribute/training_loops
"""
from pathlib import Path

import fire
import numpy as np
import sentencepiece as spm
import tensorflow as tf
import tqdm

from lm.fire_utils import only_allow_defined_args


def create_model(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    print('model', model)
    return model


@only_allow_defined_args
def main(
        run_path,
        dataset_path,
        sp_model_path,
        n_ctx=32,
        batch_size_per_replica=4,
        epochs=2,
        ):

    sp_model = spm.SentencePieceProcessor()
    sp_model.load(sp_model_path)

    dataset_path = Path(dataset_path)
    print(f'Loading dataset from {dataset_path}')
    # TODO
    # valid_dataset = np.load(dataset_path / 'valid.npy')
    train_dataset = np.load(dataset_path / 'train.npy')
    print(f'Train dataset has {len(train_dataset):,} tokens')
    # print(f'Validation dataset has {len(valid_dataset):,} tokens')

    strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    step_tokens = n_ctx * batch_size
    train_steps_per_epoch = len(train_dataset) // step_tokens
    test_steps_per_epoch = len(train_dataset) // step_tokens

    # TODO: re-create this each epoch
    train_indices = [np.random.randint(0, len(train_dataset) - n_ctx)
                     for _ in range(len(train_dataset) // n_ctx)]
    train_contexts = [train_dataset[idx: idx + n_ctx] for idx in train_indices]

    with strategy.scope():
        train_iterator = strategy.experimental_make_numpy_iterator(
            train_contexts, batch_size, shuffle=None)
        # test_iterator = strategy.experimental_make_numpy_iterator(
        #    (test_images, test_labels), batch_size, shuffle=None)

    # Create a checkpoint directory to store the checkpoints.
    run_path = Path(run_path)
    checkpoint_path = run_path / 'checkpoints'

    with strategy.scope():
        # TODO sparse_softmax_cross_entropy_with_logits
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        # test_loss = tf.keras.metrics.Mean(name='test_loss')

        model = create_model(vocab_size=len(sp_model))
        optimizer = tf.keras.optimizers.Adam()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        def train_step(context):
            with tf.GradientTape() as tape:
                predictions = model(context, training=True)
                loss = loss_object(context[:, 1:],
                                   predictions[:, :-1])
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)

       #def test_step(inputs):
       #    images, labels = inputs
       #    predictions = model(images, training=False)
       #    t_loss = loss_object(labels, predictions)
       #    test_loss(t_loss)

        @tf.function
        def distributed_train():
            return strategy.experimental_run(train_step, train_iterator)

       #@tf.function
       #def distributed_test():
       #    return strategy.experimental_run(test_step, test_iterator)

        for epoch in range(epochs):

            # TRAIN LOOP
            # Initialize the iterator
            train_iterator.initialize()
            train_pbar = tqdm.trange(train_steps_per_epoch, desc='train')
            for _ in train_pbar:
                distributed_train()
                train_pbar.set_postfix(loss=f'{train_loss.result():.4f}')

           ## TEST LOOP
           #test_iterator.initialize()
           #for _ in tqdm.trange(test_steps_per_epoch, desc='validate'):
           #    distributed_test()

            checkpoint.save(checkpoint_path)

            print(f'epoch: {epoch + 1}, train_loss: {train_loss.result():.4f}')

            train_loss.reset_states()
            # test_loss.reset_states()


if __name__ == '__main__':
    fire.Fire(main)

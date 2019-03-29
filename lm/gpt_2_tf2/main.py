"""
Training loop, based on
https://www.tensorflow.org/alpha/tutorials/distribute/training_loops
"""
import json
from pathlib import Path
import shutil
import sys

import attr
import fire
import numpy as np
import sentencepiece as spm
import tensorflow as tf
import tqdm

from ..fire_utils import only_allow_defined_args
from .model import Model, HParams


@only_allow_defined_args
def main(
        run_path,
        dataset_path,
        sp_model_path,
        epochs=10,
        lr=2.5e-4,
        batch_size_per_replica=2,
        accum_gradients=32,  # accumulate gradients N times
        n_ctx=1024,
        n_embed=768,
        n_head=12,
        n_layer=12,
        clean=False,  # clean run folder
        log_every=1,
        validate_every=None,
        save_every=None,
        ):

    run_path = Path(run_path)
    run_path_mark = run_path / '.lm'
    if clean and run_path.exists():
        assert run_path_mark.exists()  # to avoid removing unrelated folder
        shutil.rmtree(run_path)
    run_path.mkdir(exist_ok=True, parents=True)
    run_path_mark.touch()
    checkpoint_prefix = run_path / 'checkpoints' / 'model'
    summaries_path = run_path / 'summaries'

    sp_model = spm.SentencePieceProcessor()
    sp_model.load(sp_model_path)

    hparams = HParams(
        n_vocab=len(sp_model),
        n_ctx=n_ctx,
        n_embed=n_embed,
        n_head=n_head,
        n_layer=n_layer,
    )
    params = dict(
        hparams=attr.asdict(hparams),
        argv=' '.join(sys.argv),
        epochs=epochs,
        lr=lr,
        batch_size_per_replica=batch_size_per_replica,
        accum_gradients=accum_gradients,
    )
    params_s = json.dumps(params, indent=4, sort_keys=True, ensure_ascii=False)
    print(params_s)
    (run_path / 'params.json').write_text(params_s, encoding='utf8')

    dataset_path = Path(dataset_path)
    print(f'Loading dataset from {dataset_path}')
    valid_dataset = np.load(dataset_path / 'valid.npy')[:100000]
    train_dataset = np.load(dataset_path / 'train.npy')[:1000000]
    print(f'Train dataset has {len(train_dataset):,} tokens')
    print(f'Validation dataset has {len(valid_dataset):,} tokens')

    strategy = tf.distribute.MirroredStrategy()
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    step_tokens = n_ctx * batch_size
    train_steps_per_epoch = len(train_dataset) // step_tokens
    valid_steps_per_epoch = len(valid_dataset) // step_tokens

    # TODO: re-create each epoch (or change experimental_make_numpy_iterator)
    print('Creating dataset slices')
    train_indices = [np.random.randint(0, len(train_dataset) - n_ctx)
                     for _ in range(len(train_dataset) // n_ctx)]
    train_contexts = [train_dataset[idx: idx + n_ctx] for idx in train_indices]
    valid_indices = range(0, len(valid_dataset) - n_ctx, n_ctx)
    valid_contexts = [valid_dataset[idx: idx + n_ctx] for idx in valid_indices]

    # Loss is scaled by the number of replicas, as gradients are summed
    loss_fn = lambda labels, logits: \
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)) / strategy.num_replicas_in_sync

    step = 0
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    train_summary_writer = tf.summary.create_file_writer(
        str(summaries_path / 'train'))
    valid_summary_writer = tf.summary.create_file_writer(
        str(summaries_path / 'valid'))

    with strategy.scope():
        print('Creating dataset iterators')
        # FIXME this takes forever
        train_iterator = strategy.experimental_make_numpy_iterator(
            train_contexts, batch_size, shuffle=None)
        valid_iterator = strategy.experimental_make_numpy_iterator(
            valid_contexts, batch_size, shuffle=None)

        print('Creating model')
        model = Model(hparams)
        optimizer = tf.optimizers.Adam(lr)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        def train_step(context):
            context = tf.cast(context, tf.int32)
            import time
            t0 = time.time()
            loss = None
            with tf.GradientTape() as tape:
                for i in range(accum_gradients):
                    t1 = time.time()
                    print('accum_gradients', i, t1 - t0)
                    t0 = t1
                    logits = model(context)['logits']
                    l = loss_fn(context[:, 1:], logits[:, :-1])
                    train_loss(l)
                    loss = l if loss is None else l + loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        def valid_step(context):
            context = tf.cast(context, tf.int32)
            logits = model(context)['logits']
            loss = loss_fn(context[:, 1:], logits[:, :-1])
            valid_loss(loss)

        @tf.function
        def distributed_train():
            return strategy.experimental_run(train_step, train_iterator)

        @tf.function
        def distributed_validate():
            return strategy.experimental_run(valid_step, valid_iterator)

        def validate():
            valid_iterator.initialize()
            valid_loss.reset_states()
            for _ in tqdm.trange(valid_steps_per_epoch, desc='validate',
                                 leave=False, dynamic_ncols=True):
                distributed_validate()
            with valid_summary_writer.as_default():
                tf.summary.scalar('loss', valid_loss.result(),
                                  step=step * step_tokens)

        for epoch in tqdm.trange(1, epochs + 1, desc='epochs'):

            train_iterator.initialize()
            for _ in tqdm.trange(train_steps_per_epoch, desc=f'epoch {epoch}',
                                 dynamic_ncols=True):
                distributed_train()
                step += 1
                if step % log_every == 0:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss.result(),
                                          step=step * step_tokens)
                    train_loss.reset_states()
                if validate_every and step % validate_every == 0:
                    validate()
                if save_every and step % save_every == 0:
                    checkpoint.save(checkpoint_prefix)

            validate()
            checkpoint.save(checkpoint_prefix)


def fire_main():
    fire.Fire(main)

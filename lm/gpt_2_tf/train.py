"""
Based on https://github.com/nshepperd/gpt-2/blob/finetuning/train.py
"""
import json
from pathlib import Path
import sys
import shutil
from typing import List, Tuple

import fire
import numpy as np
import matplotlib.pyplot as plt
import sentencepiece as spm
import tensorflow as tf
import tqdm

from . import model, sample
from lm.data import END_OF_TEXT
from lm.fire_utils import only_allow_defined_args


def main():
    return fire.Fire(train)


@only_allow_defined_args
def train(
        run_path,
        dataset_path,
        sp_model_path,
        *,
        batch_size,
        lr=1e-3,
        epochs=10,
        sample_length=None,
        sample_num=1,
        sample_every=1000,
        restore_from=None,  # latest by default, or "path/model-STEP"
        save_every=1000,
        log_every=20,
        config='default',
        accum_gradients=1,  # accumulate gradients N times
        find_lr=False,  # instead of normal training, run lr range finder
        validate=False,  # instead of training, run validation and exit
        clean=False,  # clean run folder
        # override hparams from config
        n_ctx=None,
        n_embd=None,
        n_head=None,
        n_layer=None,
        ):

    sp_model = spm.SentencePieceProcessor()
    sp_model.load(sp_model_path)

    run_path = Path(run_path)
    if clean and run_path.exists():
        extra_names = {
            p.name for p in run_path.iterdir()
            if not (
               p.name in {'checkpoints', 'samples', 'summaries', 'params.json'}
               or p.name.startswith('find-lr')
        )}
        assert not extra_names, extra_names
        shutil.rmtree(run_path)
    run_path.mkdir(exist_ok=True, parents=True)
    checkpoints_path = run_path / 'checkpoints'
    samples_path = run_path / 'samples'
    summaries_path = run_path / 'summaries'
    dataset_path = Path(dataset_path)

    hparams = model.HPARAMS[config]
    hparams.n_vocab = len(sp_model)
    if n_ctx is not None: hparams.n_ctx = n_ctx
    n_ctx = hparams.n_ctx
    if n_embd is not None: hparams.n_embd = n_embd
    if n_head is not None: hparams.n_head = n_head
    if n_layer is not None: hparams.n_layer = n_layer
    del n_layer, n_embd, n_head
    params_text = json.dumps(dict(
        hparams=hparams.values(),
        dataset_path=str(dataset_path),
        sp_model_path=sp_model_path,
        batch_size=batch_size,
        accum_gradients=accum_gradients,
        lr=lr,
        epochs=epochs,
        restore_from=str(restore_from),
        argv=sys.argv,
    ), indent=4, sort_keys=True)
    print(params_text)
    if not (validate or find_lr):
        (run_path / 'params.json').write_text(params_text)

    if sample_length is None:
        sample_length = n_ctx - 1
    elif sample_length > n_ctx:
        raise ValueError(
            f'Can\'t get samples longer than window size: {n_ctx}')
    step_tokens = n_ctx * batch_size * accum_gradients

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        output = model.model(hparams=hparams, X=context)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        summaries_path.mkdir(exist_ok=True, parents=True)
        summary_writer = tf.summary.FileWriter(
            summaries_path / 'train', sess.graph)

        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=sample_length,
            context=context,
            batch_size=batch_size,
            temperature=1.0,
            top_k=40)

        train_vars = tf.trainable_variables()
        learning_rate = tf.placeholder(tf.float32, name='lr')
        opt = tf.train.AdamOptimizer(learning_rate)
        accum_gradients = max(accum_gradients, 1)
        if accum_gradients > 1:
            train_op, zero_ops, accum_ops = \
                _accum_gradients_ops(train_vars, opt, loss)
        else:
            train_op = opt.minimize(loss, var_list=train_vars)

        saver = tf.train.Saver(
            var_list=train_vars,
            max_to_keep=2,
            keep_checkpoint_every_n_hours=4)
        sess.run(tf.global_variables_initializer())

        if restore_from or checkpoints_path.exists():
            if restore_from is None:
                restore_from = tf.train.latest_checkpoint(checkpoints_path)
            print(f'Restoring from {restore_from}')
            saver.restore(sess, restore_from)

        print(f'Loading dataset from {dataset_path}')
        valid_dataset = np.load(dataset_path / 'valid.npy')
        print(f'Validation dataset has {len(valid_dataset):,} tokens')

        step = 1
        step_path = checkpoints_path / 'step'
        if step_path.exists():
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            step = int(step_path.read_text()) + 1

        def save():
            if find_lr:
                return
            checkpoints_path.mkdir(exist_ok=True, parents=True)
            saver.save(sess, checkpoints_path / 'model', global_step=step)
            step_path.write_text(str(step) + '\n')

        def write_summaries(**kwargs):
            summary = tf.Summary()
            for k, v in kwargs.items():
                summary.value.add(tag=k, simple_value=v)
            summary_writer.add_summary(summary, step * step_tokens)

        def generate_samples():
            context_tokens = [sp_model.PieceToId(END_OF_TEXT)]
            all_text = []
            index = 0
            while index < sample_num:
                out = sess.run(
                    tf_sample,
                    feed_dict={context: batch_size * [context_tokens]})
                for i in range(min(sample_num - index, batch_size)):
                    text = sp_model.DecodeIds(list(map(int, out[i])))
                    text = f'======== SAMPLE {index + 1} ========\n{text}\n'
                    all_text.append(text)
                    index += 1
            samples_path.mkdir(exist_ok=True, parents=True)
            (samples_path / f'samples-{step}.txt').write_text(
                '\n'.join(all_text))

        def validation():
            # TODO use more context here
            loss_values = [
                sess.run(loss, feed_dict={context: batch})
                for batch in _valid_batch_generator(
                    valid_dataset, batch_size=batch_size, n_ctx=n_ctx)]
            return np.mean(loss_values)

        if validate:
            print('Validating...')
            loss_value = validation()
            print(f'Validation loss: {loss_value:.4f}')
            return

        train_dataset = np.load(dataset_path / 'train.npy')
        print(f'Train dataset has {len(train_dataset):,} tokens')
        epoch_size = len(train_dataset) // step_tokens

        def train_step():
            batch = _gen_batch(
                train_dataset,
                n_ctx=n_ctx,
                batch_size=batch_size * accum_gradients,
            )
            if accum_gradients > 1:
                sess.run(zero_ops)
                loss_value = 0.
                for i in range(accum_gradients):
                    mini_batch = batch[i * batch_size: (i + 1) * batch_size]
                    *_, mb_loss_value = sess.run(
                        accum_ops + [loss], feed_dict={context: mini_batch})
                    loss_value += mb_loss_value / accum_gradients
                sess.run(train_op, feed_dict={learning_rate: lr})
            else:
                _, loss_value = sess.run(
                    [train_op, loss],
                    feed_dict={context: batch, learning_rate: lr})
            if step % log_every == 0 and not find_lr:
                write_summaries(loss=loss_value, learning_rate=lr)
            return loss_value

        if find_lr:
            lr = 1e-6
        max_lr = 10
        lr_multiplier = 1.25
        lr_data = []
        find_lr_path = run_path / f'find-lr-{step}.png'

        print('Training...')
        avg_loss = (0.0, 0.0)
        try:
            for epoch in tqdm.trange(1, epochs + 1, desc='epoch'):
                epoch_pbar = tqdm.trange(epoch_size, desc=f'epoch {epoch}')
                for _ in epoch_pbar:

                    if step % save_every == 0:
                        save()
                        valid_loss = validation()
                        write_summaries(valid_loss=valid_loss)
                    if step % sample_every == 0:
                        generate_samples()

                    lv = train_step()
                    step += 1
                    if find_lr:
                        lr *= lr_multiplier
                        if lr > max_lr or lr_data and lv > 2 * lr_data[0][1]:
                            _plot_find_lr_data(lr_data, find_lr_path)
                            return
                        lr_data.append((lr, lv))

                    avg_loss = (avg_loss[0] * 0.99 + lv,
                                avg_loss[1] * 0.99 + 1.0)
                    avg = avg_loss[0] / avg_loss[1]
                    epoch_pbar.set_postfix({
                        'step': step,
                        'loss': f'{lv:.2f}',
                        'avg': f'{avg:.2f}',
                    })
            save()

        except KeyboardInterrupt:
            print('Interrupted, saving')
            save()
            sys.exit(1)


def _gen_batch(dataset: np.ndarray, n_ctx: int, batch_size: int):
    indices = [np.random.randint(0, len(dataset) - n_ctx)
               for _ in range(batch_size)]
    return [dataset[idx : idx + n_ctx] for idx in indices]


def _accum_gradients_ops(train_vars, opt, loss):
    # https://stackoverflow.com/a/46773161/217088
    accum_vars = [tf.Variable(tf.zeros_like(v.initialized_value()),
                              trainable=False)
                  for v in train_vars]
    zero_ops = [v.assign(tf.zeros_like(v)) for v in accum_vars]
    gvs = opt.compute_gradients(loss, train_vars)
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
    train_op = opt.apply_gradients(
        [(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])
    return train_op, zero_ops, accum_ops


def _plot_find_lr_data(lr_data: List[Tuple[float, float]], path: Path):
    plt.figure(figsize=(12, 6))
    plt.plot([lr for lr, _ in lr_data], [lv for _, lv in lr_data])
    plt.xscale('log')
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.savefig(path)
    print(f'Saved lr range test to {path}')
    # TODO - save results to json as well, to be able to re-plot


def _valid_batch_generator(dataset, *, batch_size: int, n_ctx: int):
    start_indices = range(0, len(dataset) - n_ctx, n_ctx)
    return _batch_it(
        (dataset[start_idx: start_idx + n_ctx] for start_idx in tqdm.tqdm(
            start_indices, desc='validation', leave=False)),
        batch_size=batch_size)


def _batch_it(it, batch_size: int):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    # last is dropped

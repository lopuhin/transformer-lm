"""
Based on https://github.com/nshepperd/gpt-2/blob/finetuning/train.py
"""
import json
from pathlib import Path
import sys
import shutil

import fire
import numpy as np
import sentencepiece as spm
import tensorflow as tf
import tqdm

from . import model, sample
from lm.data import END_OF_TEXT


def main():
    return fire.Fire(train)


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
        restore_from=None,  # checkpoint path, or from latest by default
        save_every=1000,
        log_every=20,
        config='default',
        accum_gradients=1,  # accumulate gradients N times
        clean=False,
        ):

    sp_model = spm.SentencePieceProcessor()
    sp_model.load(sp_model_path)

    run_path = Path(run_path)
    if clean:
        extra_names = {p.name for p in run_path.iterdir()} - {
            'checkpoints', 'samples', 'summaries', 'params.json'}
        assert not extra_names, extra_names
        shutil.rmtree(run_path)
    run_path.mkdir(exist_ok=True, parents=True)
    checkpoints_path = run_path / 'checkpoints'
    samples_path = run_path / 'samples'
    summaries_path = run_path / 'summaries'
    dataset_path = Path(dataset_path)
    train_path = dataset_path / 'train.npy'
    if checkpoints_path.exists() and restore_from is None:
        restore_from = checkpoints_path

    hparams = model.HPARAMS[config]
    hparams.n_vocab = len(sp_model)
    (run_path / 'params.json').write_text(json.dumps(dict(
        hparams=hparams.values(),
        dataset_path=str(dataset_path),
        sp_model_path=sp_model_path,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        restore_from=str(restore_from),
        argv=sys.argv,
    ), indent=4, sort_keys=True))

    if sample_length is None:
        sample_length = hparams.n_ctx - 1
    elif sample_length > hparams.n_ctx:
        raise ValueError(
            f'Can\'t get samples longer than window size: {hparams.n_ctx}')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        output = model.model(hparams=hparams, X=context)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))
        tf.summary.scalar('loss', loss)

        summaries = tf.summary.merge_all()
        summaries_path.mkdir(exist_ok=True, parents=True)
        train_writer = tf.summary.FileWriter(
            summaries_path / 'train', sess.graph)

        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=sample_length,
            context=context,
            batch_size=batch_size,
            temperature=1.0,
            top_k=40)

        train_vars = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(lr)
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

        if restore_from:
            print(f'Restoring from {restore_from}')
            ckpt = tf.train.latest_checkpoint(restore_from)
            print(f'Loading checkpoint {ckpt}')
            saver.restore(sess, ckpt)

        print(f'Loading dataset {train_path}')
        dataset = np.load(train_path)
        print(f'Dataset has {len(dataset):,} tokens')
        print('Training...')

        step = 1
        step_path = checkpoints_path / 'step'
        if step_path.exists():
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            step = int(step_path.read_text()) + 1

        step_tokens = hparams.n_ctx * batch_size * accum_gradients
        epoch_size = len(dataset) // step_tokens

        def save():
            checkpoints_path.mkdir(exist_ok=True, parents=True)
            saver.save(sess, checkpoints_path / 'model', global_step=step)
            step_path.write_text(str(step) + '\n')

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

        def train_step():
            batch = _gen_batch(
                dataset,
                n_ctx=hparams.n_ctx,
                batch_size=batch_size * accum_gradients,
            )
            if accum_gradients > 1:
                sess.run(zero_ops)
                for i in range(accum_gradients):
                    mini_batch = batch[i * batch_size: (i + 1) * batch_size]
                    ops = accum_ops
                    if i == 0:
                        ops = ops + [loss, summaries]
                    res = sess.run(ops, feed_dict={context: mini_batch})
                    if i == 0:
                        loss_value, summary = res[-2:]
                sess.run(train_op)
            else:
                _, loss_value, summary = sess.run(
                    [train_op, loss, summaries],
                    feed_dict={context: batch})
            if step % log_every == 0:
                train_writer.add_summary(summary, step * step_tokens)
            return loss_value

        avg_loss = (0.0, 0.0)
        try:
            for epoch in tqdm.trange(1, epochs + 1, desc='epoch'):
                epoch_pbar = tqdm.trange(epoch_size, desc=f'epoch {epoch}')
                for _ in epoch_pbar:

                    if step % save_every == 0:
                        save()
                    if step % sample_every == 0:
                        generate_samples()

                    lv = train_step()
                    step += 1

                    avg_loss = (avg_loss[0] * 0.99 + lv,
                                avg_loss[1] * 0.99 + 1.0)
                    avg = avg_loss[0] / avg_loss[1]
                    epoch_pbar.set_postfix({
                        'step': step,
                        'loss': f'{lv:.2f}',
                        'avg': f'{avg:.2f}',
                    })

        except KeyboardInterrupt:
            print('Interrupted, saving')
            save()


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

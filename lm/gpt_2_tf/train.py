"""
Based on https://github.com/nshepperd/gpt-2/blob/finetuning/train.py
"""
from pathlib import Path

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
        epochs=10,
        seed=None,
        sample_length=None,
        sample_num=1,
        sample_every=1000,
        restore_from=None,  # checkpoint path, or from latest by default
        save_every=1000,
        config='default',
        ):

    sp_model = spm.SentencePieceProcessor()
    sp_model.load(sp_model_path)

    run_path = Path(run_path)
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

    if sample_length is None:
        sample_length = hparams.n_ctx - 1
    elif sample_length > hparams.n_ctx:
        raise ValueError(
            f'Can\'t get samples longer than window size: {hparams.n_ctx}')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
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

        train_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        opt = tf.train.AdamOptimizer().minimize(loss, var_list=train_vars)

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

        avg_loss = (0.0, 0.0)
        try:
            for epoch in tqdm.trange(1, epochs + 1, desc='epoch'):
                epoch_size = len(dataset) // hparams.n_ctx
                epoch_pbar = tqdm.trange(epoch_size, desc=f'epoch {epoch}')
                for _ in epoch_pbar:

                    if step % save_every == 0:
                        save()
                    if step % sample_every == 0:
                        generate_samples()

                    batch = _gen_batch(
                        dataset, n_ctx=hparams.n_ctx, batch_size=batch_size)
                    _, lv, summary = sess.run(
                        [opt, loss, summaries], feed_dict={context: batch})
                    train_writer.add_summary(summary, step)
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
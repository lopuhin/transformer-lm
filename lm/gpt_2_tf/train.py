"""
Based on https://github.com/nshepperd/gpt-2/blob/finetuning/train.py
"""
from pathlib import Path
import time

import fire
import numpy as np
import sentencepiece as spm
import tensorflow as tf

from . import model, sample
from lm.data import END_OF_TEXT


@fire.Fire
def main(
        run_path,
        dataset_path,
        sp_model_path,
        *,
        seed=None,
        batch_size=1,
        sample_length=None,
        sample_num=1,
        sample_every=100,
        restore_from=None,  # 'latest' or checkpoint path
        save_every=1000,
        config='default',
        ):
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(sp_model_path)

    run_path = Path(run_path)
    run_path.mkdir(exist_ok=True, parents=True)
    checkpoint_path = run_path / 'checkpoint'
    samples_path = run_path / 'samples'
    dataset_path = Path(dataset_path)
    train_path = dataset_path / 'train.npy'

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

        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=sample_length,
            context=context,
            batch_size=batch_size,  # TODO try min(sample_num, batch_size)
            temperature=1.0,
            top_k=40)

        train_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        opt = tf.train.AdamOptimizer().minimize(loss, var_list=train_vars)

        saver = tf.train.Saver(
            var_list=train_vars,
            max_to_keep=4,
            keep_checkpoint_every_n_hours=1)
        sess.run(tf.global_variables_initializer())

        if restore_from is not None:
            ckpt = tf.train.latest_checkpoint(
                checkpoint_path if restore_from == 'latest' else restore_from)
            print('Loading checkpoint', ckpt)
            saver.restore(sess, ckpt)

        print('Loading dataset...')
        dataset = np.load(train_path)
        print(f'Dataset has {len(dataset):,} tokens')
        print('Training...')

        counter = 1
        counter_path = checkpoint_path / 'counter'
        if counter_path.exists():
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            counter = int(counter_path.read_text()) + 1

        def save():
            checkpoint_path.mkdir(exist_ok=True, parents=True)
            print(f'Saving at {counter:,}')
            saver.save(sess, checkpoint_path / 'model', global_step=counter)
            counter_path.write_text(str(counter) + '\n')

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
                    print(text)

            samples_path.mkdir(exist_ok=True, parents=True)
            (samples_path / f'samples-{counter}.txt').write_text(
                '\n'.join(all_text))

        avg_loss = (0.0, 0.0)
        start_time = time.time()

        try:
            while True:
                if counter % save_every == 0:
                    save()
                if counter % sample_every == 0:
                    generate_samples()

                indices = [np.random.randint(0, len(dataset) - hparams.n_ctx)
                           for _ in range(batch_size)]
                batch = [dataset[idx : idx + hparams.n_ctx]
                         for idx in indices]

                _, lv = sess.run((opt, loss), feed_dict={context: batch})

                avg_loss = (avg_loss[0] * 0.99 + lv, avg_loss[1] * 0.99 + 1.0)
                elapsed = time.time() - start_time
                avg = avg_loss[0] / avg_loss[1]
                print(f'[{counter} | {elapsed:0f}] '
                      f'loss={lv:.2f} avg={avg:.2f}')

                counter += 1
        except KeyboardInterrupt:
            print('Interrupted, saving')
            save()

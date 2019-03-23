from functools import partial
from pathlib import Path
from typing import Dict

import fire
import numpy as np
import sentencepiece as spm
import tensorflow as tf

from ..fire_utils import only_allow_defined_args


class Model(tf.Module):
    def __init__(self, n_vocab, name=None):
        super().__init__(name=name)
        embedding_size = 64
        self.emb = tf.Variable(
            tf.random.uniform([n_vocab, embedding_size]), name='emb')
        self.w = tf.Variable(
            tf.random.normal([1, embedding_size, n_vocab], stddev=0.05),
            name='w')
        self.b = tf.Variable(tf.zeros([n_vocab]), name='b')

    def __call__(self, x):
        print('x', x.shape)
        h = tf.nn.embedding_lookup(self.emb, x)
        print('h', h.shape)
        y = tf.nn.conv1d(h, self.w, 1, 'SAME') + self.b
        print('y', y.shape)
        return y


def estimator_spec(
        features: tf.Tensor,
        labels,
        mode:   tf.estimator.ModeKeys,
        params: Dict = None,
        config: tf.estimator.RunConfig = None,
        ) -> tf.estimator.EstimatorSpec:
    print(f'estimator_spec mode {mode}')
    print(f'estimator_spec features {features}')
    assert labels is None
    model = Model(n_vocab=params['n_vocab'], name='model')
    optimizer = tf.optimizers.Adam(learning_rate=0.1)
    print('trainable_variables', model.trainable_variables)
    # TODO do we need tf.function somewhere?

    with tf.GradientTape() as tape:
        logits = model(features)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=features[:, 1:],
                logits=logits[:, :-1]))

    gradients = tape.gradient(loss, model.trainable_variables)
    train_op = optimizer.apply_gradients(
        zip(gradients, model.trainable_variables))

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
    )


@only_allow_defined_args
def main(
        dataset_path,
        sp_model_path,
        run_path=None,
        ):

    sp_model = spm.SentencePieceProcessor()
    sp_model.load(sp_model_path)

    dataset_path = Path(dataset_path)
    print(f'Loading dataset from {dataset_path}')
    valid_dataset = np.load(dataset_path / 'valid.npy')
    print(f'Validation dataset has {len(valid_dataset):,} tokens')
    train_dataset = np.load(dataset_path / 'train.npy')
    print(f'Train dataset has {len(train_dataset):,} tokens')

    # TODO handle config
    n_ctx = 32
    batch_size = 4

    estimator = tf.estimator.Estimator(
        estimator_spec,
        model_dir=run_path,
        params={
            'n_vocab': len(sp_model),
        },
    )
    estimator.train(
        input_fn=partial(_gen_batch, train_dataset, n_ctx, batch_size),
        steps=5,
    )
    import IPython; IPython.embed()


def _gen_batch(dataset: np.ndarray, n_ctx: int, batch_size: int):
    indices = [np.random.randint(0, len(dataset) - n_ctx)
               for _ in range(batch_size)]
    features = [dataset[idx : idx + n_ctx] for idx in indices]
    return np.array(features, dtype=np.int32), None


def fire_main():
    fire.Fire(main)

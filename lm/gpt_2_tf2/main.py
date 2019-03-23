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
        h = tf.nn.embedding_lookup(self.emb, x)
        y = tf.nn.conv1d(h, self.w, 1, 'SAME') + self.b
        return y


def estimator_spec(
        features: tf.Tensor,
        labels,
        mode:   tf.estimator.ModeKeys,
        params: Dict = None,
        config: tf.estimator.RunConfig = None,
        ) -> tf.estimator.EstimatorSpec:
    print(f'estimator_spec mode {mode}')
    # print(f'estimator_spec features {features}')
    assert labels is None
    model = Model(n_vocab=params['n_vocab'], name='model')
    optimizer = tf.compat.v1.train.AdamOptimizer()
    global_step = tf.compat.v1.train.get_or_create_global_step()
    # print('trainable_variables', model.trainable_variables)
    # TODO do we need tf.function somewhere?
    # TODO gradient accumulation (maybe check 0128fcb again)

    logits = model(features)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=features[:, 1:],
            logits=logits[:, :-1]))

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(loss, global_step),
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

    # TODO handle hyperparameters config
    n_ctx = 32
    batch_size = 4

    run_config = tf.estimator.RunConfig(
        save_summary_steps=10,
    )
    estimator = tf.estimator.Estimator(
        estimator_spec,
        model_dir=run_path,
        params={
            'n_vocab': len(sp_model),
        },
        config=run_config,
    )
    estimator.train(
        input_fn=partial(_train_batch, train_dataset, n_ctx, batch_size),
        steps=100,
    )
    valid_iter = _valid_iter(valid_dataset[:1000], batch_size=batch_size, n_ctx=n_ctx)
    eval_result = estimator.evaluate(lambda: next(valid_iter), steps=100)
    import IPython; IPython.embed()


def _train_batch(dataset: np.ndarray, n_ctx: int, batch_size: int):
    print('************************* _train_batch')
    tf.print('**********************fo00')
    indices = [np.random.randint(0, len(dataset) - n_ctx)
               for _ in range(batch_size)]
    features = [dataset[idx : idx + n_ctx] for idx in indices]
    return np.array(features, dtype=np.int32), None


def _valid_iter(dataset, *, batch_size: int, n_ctx: int):
    start_indices = range(0, len(dataset) - n_ctx, n_ctx)
    return _batch_it(
        (dataset[start_idx: start_idx + n_ctx] for start_idx in start_indices),
        batch_size=batch_size)


def _batch_it(it, batch_size: int):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == batch_size:
            print('yielding batch')
            yield np.array(batch, dtype=np.int32), None
            batch = []
    # last is dropped


def fire_main():
    fire.Fire(main)

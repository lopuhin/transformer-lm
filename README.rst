Training GPT-2 transformer language model with sentencepiece tokenizer
======================================================================

.. image:: https://img.shields.io/travis/lopuhin/transformer-lm/master.svg
   :target: https://travis-ci.org/lopuhin/transformer-lm
   :alt: Build Status

Training GPT-2 transformer language model on your own corpora
with `sentencepiece <https://github.com/google/sentencepiece>`_ tokenization.

This repo contains a PyTorch implementation of GPT-2, which support multi-GPU
training.
It also contains a TensorFlow implementation in ``lm/gpt_2_tf``,
but it is not developed any more. They share the same data preparation scripts.
TF training command is ``gpt-2-tf-train`` and needs TensorFlow 1.13.
Documentation below is for PyTorch version.

.. contents::

Installation
------------

Python 3.6+ is required. Working in a virtualenv is assumed below.
`Install <https://pytorch.org/get-started/locally/>`__
appropriate version of pytorch first (e.g. ``pip install torch``), and then::

    pip install -r requirements.txt
    python setup.py develop


Usage
-----

Instructions are below. See also ``test/test_shakespeare.sh``
for a complete pipeline demo on a small corpus (takes a minute on a CPU).

Prepare data for training
+++++++++++++++++++++++++

Corpus format: a directory with top-level ``train``, ``valid`` and ``test``
folders. Each top-level folder may contain sub-folders. Inside them,
there must be utf-8 encoded text files with ``.txt`` extension.

The commands to train sentencepiece model and encode the corpus support
multiple corpora,
in below examples we assume they can be listed as ``data/corpora-*``.

1. Train sentencepiece model (``sp-text.txt`` can be removed after running).
   This can consume a large amount of memory, adjust sentencepiece arguments
   as advised if needed
   (this is not supported in the ``sp-train`` command directly)::

    sp-train data/corpora-* sp-text.txt sp-model

2. Encode corpora, producing numpy files::

    sp-encode data/corpora-* sp-model.model data/encoded


Training
++++++++

Example command::

    gpt-2 run-root data/encoded sp-model.model

``run-root`` would contain model checkpoints and json-lines logs,
which can be plotted in a jupyter notebook with
``json_log_plots.plot("run-root")``, with number of tokens seen on the X axis.

Default hyperparameters correspond to released "small" GPT-2 model.

When multiple GPUs are available, they would be used for training with the
help of ``torch.distributed``.

If the path exists and ``--clean`` key is NOT passed, training would be resumed.
Note that all parameters still need to be specified and
model parameters need to match.

Notes on training parameters:

- ``--batch-size`` is per-GPU, so you don't need to re-tune it when changing
  number of GPUs, just use max that fits into memory.
- ``--g-accum-gradients`` is the global number of gradient accumulations,
  it must be divisible by the number of GPUs. Effective global batch size is
  always ``batch_size * g_accum_gradients``.
- ``--lr`` does not need to be changed when changing
  ``--batch-size`` or ``--g-accum-gradients`` or number of GPUs
  or ``--n-ctx``: loss is already scaled appropriately.


Inference
+++++++++

Example command::

    gpt-2-gen run-root "Artificial intelligence"

``run-root`` would contain model checkpoints
``"Artificial intelligence"`` is the text prefix used as a starting point for generating tokens

Notes on inference parameters:

- ``--tokens-to-generate``: number of tokens to generate, default is 42
- ``--top-k``: number of token candidates to generate for each position (beam width), default is 8.


License & credits
-----------------

License is MIT.

TensorFlow GPT-2 model is taken from
https://github.com/openai/gpt-2/blob/master/src/model.py
and TensorFlow GPT-2 training code is based on
https://github.com/nshepperd/gpt-2/blob/finetuning/train.py

PyTorch port is based on original OpenAI code.

Test Shakespeare corpus under ``tests/shakespeare``
is from http://shakespeare.mit.edu under public domain.

See also OpenAI GPT-2
`paper <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`_
and `blog <https://openai.com/blog/better-language-models/>`_.

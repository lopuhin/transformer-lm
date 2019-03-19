Transformer language model with sentencepiece tokenizer
=======================================================

Training transformer language models (currently only GPT-2) on your own corpora
with `sentencepiece <https://github.com/google/sentencepiece>`_ tokenization.

.. contents::

Installation
------------

Python 3.6+ is required. Working in a virtualenv is assumed below.
`Install <https://www.tensorflow.org/install/pip>`_
appropriate version of Tensorflow 1.13 first, and then::

    pip install -r requirements.txt
    python setup.py develop


Usage
-----

Instructions are below. See also ``test/test_shakespeare.sh``
for a complete pipeline demo on a small corpus (takes a few minutes on CPU).

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

Currently training of OpenAI GPT-2 model is supported, example command::

    gpt-2-tf-train \
        run-root data/encoded sp-model.model \
        --batch-size 32 --sample-num 4 --config small

``run-root`` would contain Tensorboard logs,
model checkpoints and generated samples.

License & credits
-----------------

License is MIT.

GPT-2 model is taken from
https://github.com/openai/gpt-2/blob/master/src/model.py
and GPT-2 training code is based on
https://github.com/nshepperd/gpt-2/blob/finetuning/train.py

Test Shakespeare corpus under ``tests/shakespeare``
is from http://shakespeare.mit.edu under public domain.

See also OpenAI GPT-2
`paper <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`_
and `blog <https://openai.com/blog/better-language-models/>`_.

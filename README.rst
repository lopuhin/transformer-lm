Transformer language model with sentencepiece tokenizer
=======================================================

Installation
------------

Python 3.6+ is required. Working in a virtualenv is assumed below::

    pip install -r requirements.txt
    python setup.py develop


Usage
-----

Corpus format: a directory with top-level ``train``, ``valid`` and ``test``
folders. Each top-level folder may contain sub-folders. Inside them,
there must be utf-8 encoded text files with ``.txt`` extension.

In all commands, multiple corpora are supported (the model will be trained
and validated on all corpora).

License
-------

License is MIT.
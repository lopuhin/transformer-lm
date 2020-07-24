#!/usr/bin/env bash
# Test preparation and training on a very small corpus using char tokenizer.
# To be run from repo root.

set -ev

char-train \
    tests/shakespeare/ \
    tests/shakespeare/chars.json \
    --max-vocab-size 60

tokenize-corpus \
    tests/shakespeare/ \
    tests/shakespeare/chars.json \
    tests/shakespeare-char-encoded \
    --chunk-size 10000

gpt-2 \
    tests/shakespeare-test-run-char/ \
    tests/shakespeare-char-encoded/ \
    tests/shakespeare/chars.json \
    --batch-size 8 \
    --g-accum-gradients 2 \
    --n-ctx 48 \
    --n-embed 32 \
    --n-hidden 16 \
    --n-head 4 \
    --n-layer 3 \
    --epochs 1 \
    --log-every 2 \
    --save-every 50 \
    --validate-every 100 \
    --sample-sentences \
    --clean

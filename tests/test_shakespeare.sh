#!/usr/bin/env bash
# Test preparation and training on a very small corpus
# To be run from repo root

set -ev

sp-train \
    tests/shakespeare/ \
    tests/shakespeare/sp-text.txt \
    tests/shakespeare/sp-model \
    --vocab-size 2000

sp-encode \
    tests/shakespeare/ \
    tests/shakespeare/sp-model.model \
    tests/shakespeare-encoded

gpt-2 \
    tests/shakespeare-test-run/ \
    tests/shakespeare-encoded/ \
    tests/shakespeare/sp-model.model \
    --batch-size 8 \
    --g-accum-gradients 2 \
    --n-ctx 48 \
    --n-embed 32 \
    --n-hidden 16 \
    --n-head 4 \
    --n-layer 3 \
    --epochs 2 \
    --log-every 2 \
    --save-every 50 \
    --validate-every 100 \
    --clean

# resume training with slightly adjusted settings
gpt-2 \
    tests/shakespeare-test-run/ \
    tests/shakespeare-encoded/ \
    tests/shakespeare/sp-model.model \
    --batch-size 8 \
    --g-accum-gradients 1 \
    --n-ctx 48 \
    --n-embed 32 \
    --n-hidden 16 \
    --n-head 4 \
    --n-layer 3 \
    --validate-every 100 \
    --sample-sentences \
    --epochs 3

# run only validation
gpt-2 \
    tests/shakespeare-test-run/ \
    tests/shakespeare-encoded/ \
    tests/shakespeare/sp-model.model \
    --batch-size 8 \
    --n-ctx 48 \
    --n-embed 32 \
    --n-hidden 16 \
    --n-head 4 \
    --n-layer 3 \
    --only-validate

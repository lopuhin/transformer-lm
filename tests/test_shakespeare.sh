#!/usr/bin/env bash
# Test preparation and training on a very small corpus
# To be run from repo root

set -ev

sp-train \
    tests/shakespeare/ \
    tests/shakespeare/sp-text.txt \
    tests/shakespeare/sp-model \
    --vocab-size 4000

sp-encode \
    tests/shakespeare/ \
    tests/shakespeare/sp-model.model \
    tests/shakespeare-encoded

gpt-2-tf-train \
    tests/shakespeare-test-run/ \
    tests/shakespeare-encoded/ \
    tests/shakespeare/sp-model.model \
    --batch-size 4 \
    --accum-gradients 2 \
    --config tiny \
    --epochs 1 \
    --log-every 2 \
    --save-every 50 \
    --sample-every 50 \
    --sample-num 4 \
    --clean

gpt-2-tf-train \
    tests/shakespeare-test-run/ \
    tests/shakespeare-encoded/ \
    tests/shakespeare/sp-model.model \
    --batch-size 4 \
    --accum-gradients 2 \
    --config tiny \
    --find-lr

gpt-2-tf-train \
    tests/shakespeare-test-run/ \
    tests/shakespeare-encoded/ \
    tests/shakespeare/sp-model.model \
    --batch-size 4 \
    --config tiny \
    --validate

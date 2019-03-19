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

rm -rf tests/shakespeare-test-run/  # to avoid loading the old model
gpt-2-tf-train \
    tests/shakespeare-test-run/ \
    tests/shakespeare-encoded/ \
    tests/shakespeare/sp-model.model \
    --batch-size 4 \
    --config small \
    --epochs 2 \
    --log-every 2 \
    --sample-every 10 \
    --sample-num 2

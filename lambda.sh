#!/usr/bin/env bash

mkdir -p lambda
rm lambda/*
# this needs to be run on Linux to fetch Linux wheels
pip wheel -r requirements.lambda.txt  -w lambda
python setup.py bdist_wheel -d lambda
cp lambda.py lambda
zip -r lambda.zip lambda
ls -lh lambda.zip

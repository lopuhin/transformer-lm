import logging
from pathlib import Path
import subprocess
import shutil
import sys
from typing import List

import boto3


logger = logging.getLogger('lambda')
logger.setLevel(logging.INFO)


def install(packages: List[str]):
    logging.info(f'Installing packages {packages}')
    subprocess.call([sys.executable, '-m', 'pip', 'install'] + packages)
    logging.info('Packages installed')


install(list(map(str, Path('.').glob('*.whl'))))


logging.info('Importing ModelWrapper')
from lm.inference import ModelWrapper

MODEL_S3_BUCKET = 'transformer-lm'
MODEL_S3_PREFIX = 'runs/run-nembed768-nlayer8-bs2-ag2-vocab50k/'


def load_model() -> ModelWrapper:
    logging.info(f'Loading model s3://{MODEL_S3_BUCKET}/{MODEL_S3_PREFIX}')
    s3 = boto3.resource('s3')
    model_path = Path('/tmp/model/')
    model_path.mkdir(exist_ok=True)
    files = ['params.json', 'sp.model', 'model.pt']
    for f in files:
        target = model_path / f
        if not target.exists():
            logging.info('Downloading {f}')
            s3.Bucket(MODEL_S3_BUCKET).download_file(
                MODEL_S3_PREFIX + f, str(target))
    logging.info('Initializing model')
    model = ModelWrapper.load(model_path)
    logging.info('Cleaning up temporary storage')
    shutil.rmtree(model_path)
    logging.info('Model loaded, vocab')
    return model


model = load_model()


def lambda_handler(event, context):
    return str(event)

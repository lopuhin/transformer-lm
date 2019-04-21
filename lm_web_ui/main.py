import argparse
from pathlib import Path
from typing import List

import aiohttp_jinja2
from aiohttp import web
import jinja2

import lm_web_ui
from lm.inference import ModelWrapper


app = web.Application()
INITIAL_TEXT = 'Она открыла дверь на'


@aiohttp_jinja2.template('index.jinja2')
def index(request):
    text = request.query.get('text', INITIAL_TEXT)
    ctx = {'text': text}
    model: ModelWrapper = app['model']
    if request.query.get('predict_next_token'):
        next_top_k = model.get_next_top_k(tokenize(text), top_k=10)
        ctx['next_token_prediction'] = next_top_k
    elif request.query.get('score_occurred'):
        ctx['occurred_scores'] = model.get_occurred_log_probs(tokenize(text))
    return ctx


def tokenize(text: str) -> List[str]:
    return [app['model'].END_OF_TEXT] + app['model'].tokenize(text)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('model_root', type=Path)
    arg('--port', type=int, default=8000)
    arg('--host', default='localhost')
    args = parser.parse_args()

    template_root = Path(lm_web_ui.__file__).parent / 'templates'
    aiohttp_jinja2.setup(
        app, loader=jinja2.FileSystemLoader(str(template_root)))
    app['model'] = ModelWrapper.load(args.model_root)
    app.add_routes([web.get('/', index)])
    web.run_app(app, host=args.host, port=args.port)

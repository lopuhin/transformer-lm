import argparse
import base64
import csv
import json
import io
from pathlib import Path
from typing import List

import aiohttp_jinja2
from aiohttp import web
import jinja2

import lm_web_ui
from lm.inference import ModelWrapper


app = web.Application()
INITIAL_TEXT = 'Она открыла дверь на'
TITLE = 'Russian Language Model'


@aiohttp_jinja2.template('index.jinja2')
def index(request):
    text = request.query.get('text', '').strip()
    ctx = {}
    if text:
        ctx.update(dict(
            text=text,
            lines_as_separate=bool(request.query.get('lines_as_separate')),
            gen_tokens=int(request.query.get('gen_tokens')),
            gen_top_k=int(request.query.get('gen_top_k')),
            gen_top_p=float(request.query.get('gen_top_p')),
            gen_temp=float(request.query.get('gen_temp')),
        ))
    else:
        # defaults
        ctx.update(dict(
            text=INITIAL_TEXT,
            lines_as_separate=True,
            gen_tokens=50,
            gen_top_k=20,
            gen_top_p=0.0,
            gen_temp=1.0,
        ))
    model: ModelWrapper = app['model']
    ctx['title'] = TITLE

    score_words = request.query.get('score_words')
    score_tokens = request.query.get('score_tokens')
    if request.query.get('next_token'):
        handle_token_prediction(model, ctx)
    elif request.query.get('generate_text'):
        handle_text_generation(model, ctx)
    elif score_words or score_tokens:
        handle_scoring(model, ctx, score_words=score_words)
    return ctx


def handle_token_prediction(model, ctx):
    next_top_k = model.get_next_top_k(tokenize(ctx['text']), top_k=20)
    next_top_k = [[token, log_prob] for log_prob, token in next_top_k]
    ctx['next_token_prediction'] = next_top_k
    ctx['next_token_prediction_csv'] = to_csv_data_url(
        next_top_k, ['token', 'log_prob'])


def handle_text_generation(model, ctx):
    tokens = model.generate_tokens(
        tokenize(ctx['text']),
        tokens_to_generate=ctx['gen_tokens'],
        top_k=ctx['gen_top_k'],
        top_p=ctx['gen_top_p'],
        temperature=ctx['gen_temp'],
    )
    ctx['generated_text'] = model.sp_model.decode_pieces(tokens)
    # TODO paragraphs


def handle_scoring(model, ctx, score_words: bool):
    if score_words:
        scorer = model.get_occurred_word_log_probs
        unit_name = 'word'
    else:
        scorer = model.get_occurred_log_probs
        unit_name = 'token'
    if ctx['lines_as_separate']:
        texts = [t.strip() for t in ctx['text'].split('\n')]
        texts = list(filter(None, texts))
    else:
        texts = [ctx['text']]
    occurred_scores = []
    for i, t in enumerate(texts, 1):
        occurred_scores.extend(
            (i, unit, log_prob) for log_prob, unit in scorer(tokenize(t)))
    ctx['occurred_scores'] = occurred_scores
    ctx['occurred_scores_csv'] = to_csv_data_url(
        occurred_scores, ['text_no', unit_name, 'log_prob'])
    ctx['unit_name'] = unit_name


def tokenize(text: str) -> List[str]:
    tokens = [app['model'].END_OF_TEXT] + app['model'].tokenize(text)
    tokens = tokens[:app['model'].model.hparams.n_ctx]
    return tokens


def to_csv_data_url(data: List[List], header: List[str]) -> str:
    """ Return data url with base64-encoded csv.
    """
    f = io.StringIO()
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
    b64data = base64.b64encode(f.getvalue().encode('utf8')).decode('ascii')
    return f'data:text/csv;base64,{b64data}'


@aiohttp_jinja2.template('about.jinja2')
def about(request):
    ctx = {}
    ctx['title'] = TITLE
    model_params = json.loads(app['model_params'])
    model_params.pop('argv')  # too long
    ctx['model_params'] = json.dumps(model_params, indent=4, sort_keys=True)
    ctx['vocab_size'] = len(app['model'].sp_model)
    return ctx


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
    app['model_params'] = (args.model_root / 'params.json').read_text()
    app.add_routes([web.get('/', index)])
    app.add_routes([web.get('/about', about)])
    web.run_app(app, host=args.host, port=args.port)

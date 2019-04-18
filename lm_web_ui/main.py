import argparse
from pathlib import Path

import aiohttp_jinja2
from aiohttp import web
import jinja2

import lm_web_ui


@aiohttp_jinja2.template('index.jinja2')
def index(request):
    return {'name': 'Andrew', 'surname': 'Svetlov'}


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--port', type=int, default=8000)
    arg('--host', default='localhost')
    args = parser.parse_args()

    app = web.Application()
    template_root = Path(lm_web_ui.__file__).parent / 'templates'
    aiohttp_jinja2.setup(
        app,
        loader=jinja2.FileSystemLoader(str(template_root)),
    )
    app.add_routes([web.get('/', index)])
    web.run_app(app, host=args.host, port=args.port)

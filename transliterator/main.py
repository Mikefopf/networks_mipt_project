import logging
import argparse

import jinja2

import aiohttp_jinja2
from aiohttp import web
from views import index

parser = argparse.ArgumentParser(description="Networks project")
parser.add_argument('--host', help="Host to listen", default="0.0.0.0")
parser.add_argument('--port', help="Port to accept connections", default=10759)

args = parser.parse_args()

async def init_app():

    app = web.Application()

    app['websockets'] = {}

    app.on_shutdown.append(shutdown)

    aiohttp_jinja2.setup(
        app, loader=jinja2.PackageLoader('transliterator', 'templates'))

    app.router.add_get('/', index)

    return app


async def shutdown(app):
    for ws in app['websockets'].values():
        await ws.close()
    app['websockets'].clear()


async def get_app():
    """Used by aiohttp-devtools for local development."""
    import aiohttp_debugtoolbar
    app = await init_app()
    aiohttp_debugtoolbar.setup(app)
    return app


def main(args):
    logging.basicConfig(level=logging.DEBUG)

    app = init_app()
    web.run_app(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main(args)

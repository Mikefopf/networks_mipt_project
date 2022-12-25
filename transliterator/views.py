import logging

import aiohttp
import aiohttp_jinja2
from aiohttp import web
from faker import Faker

from transliterator import Transliterator
from utils import InvalidInputError

log = logging.getLogger(__name__)

TRANSLITERATOR = Transliterator(
    model='checkpoint.nemo', n_hyps=1, batch_size=256, threshold=0.0
)


def get_random_name():
    fake = Faker()
    return fake.name()

def transliterate(text):
    try:
        return TRANSLITERATOR.transliterate([text])[0][1][0][0]
    except InvalidInputError:
        return None

async def index(request):
    ws_current = web.WebSocketResponse()
    ws_ready = ws_current.can_prepare(request)
    if not ws_ready.ok:
        return aiohttp_jinja2.render_template('index.html', request, {})

    await ws_current.prepare(request)

    name = get_random_name()
    log.info('%s joined.', name)

    await ws_current.send_json({'action': 'connect', 'name': name})

    request.app['websockets'][name] = ws_current

    while True:
        msg = await ws_current.receive()

        if msg.type == aiohttp.WSMsgType.text:
            translit = transliterate(msg.data)
            if translit is not None:
                await ws_current.send_json(
                    {
                        'action': 'sent', 'name': name, 'text': msg.data, 'translit': transliterate(msg.data)
                    }
                )
            else:
                await ws_current.send_json(
                    {
                        'action': 'error', 'name': name
                    }
                )
        else:
            break

    del request.app['websockets'][name]
    log.info('%s disconnected.', name)

    return ws_current

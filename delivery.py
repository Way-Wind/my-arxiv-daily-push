"""Durable outbox: persist before deployment, acknowledge only confirmed sends."""
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests
from security import bounded_body, paper_id
from wechat_push import WeChatPusher

OUTBOX = Path('pending.json')


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)
    os.replace(temp, path)


def pending():
    if not OUTBOX.exists():
        return None
    data = json.loads(OUTBOX.read_text(encoding='utf-8'))
    return data if data['entries'] else None


def prepare(messages, papers, mode, url):
    if not url:
        raise ValueError('Preparing delivery requires a deployed page URL')
    filename = urlsplit(url).path.rsplit('/', 1)[-1]
    page = Path('pages') / filename
    digest = hashlib.sha256(page.read_bytes()).hexdigest()
    entries = []
    for index, msg in enumerate(messages):
        ids = [papers[index]['id']] if mode == 'detailed' else [p['id'] for p in papers]
        entries.append({'message': msg, 'ids': ids})
    save(OUTBOX, {'schema': 1, 'url': url, 'sha256': digest, 'entries': entries})


def verify_page(batch, base_url):
    expected = urlsplit(base_url)
    actual = urlsplit(batch['url'])
    if (expected.scheme != 'https' or actual.scheme != 'https' or
            expected.netloc != actual.netloc or actual.username or actual.password or
            actual.query or actual.fragment or
            not actual.path.startswith(expected.path.rstrip('/') + '/')):
        raise RuntimeError('Page URL outside configured HTTPS site')
    for attempt in range(5):
        try:
            response = requests.get(batch['url'], timeout=15, stream=True, allow_redirects=False)
            status = response.status_code
            if status != 200:
                response.close()
            else:
                body = bounded_body(response, 20_000_000, time.monotonic() + 15)
                if hashlib.sha256(body).hexdigest() == batch['sha256']:
                    return
        except requests.RequestException:
            pass
        if attempt < 4:
            time.sleep(5)
    raise RuntimeError('Published page is unavailable or differs from prepared page; notification stopped')


def notify(config, base_url):
    batch = pending()
    if not batch:
        return
    verify_page(batch, base_url)
    for key in ('app_id', 'app_secret', 'template_id', 'user_openid'):
        if not config.get(key):
            raise ValueError('Missing WeChat configuration')
    pusher = WeChatPusher(config['app_id'], config['app_secret'], config['template_id'], config['user_openid'])
    state = Path('last_pushed.json')
    ids = set(json.loads(state.read_text(encoding='utf-8'))['ids']) if state.exists() else set()
    while batch['entries']:
        entry = batch['entries'][0]
        confirmed = {paper_id(pid) for pid in entry['ids']}
        if not confirmed <= ids:
            if pusher.send_template(**entry['message']) is not True:
                raise RuntimeError('Message not confirmed')
            ids.update(confirmed)
            save(state, {'ids': sorted(ids)})
        batch['entries'].pop(0)
        save(OUTBOX, batch)
        if batch['entries']:
            time.sleep(2)

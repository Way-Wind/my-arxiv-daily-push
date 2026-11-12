"""Shared validation and bounded HTTP responses. Never include remote bodies in errors."""
import json
import os
import re
import time
from urllib.parse import urlsplit


class FatalServiceError(RuntimeError):
    """A failure that must not enter retry/fallback loops."""


def positive_limit(name, default, ceiling):
    value = int(os.environ.get(name) or default)
    if not 1 <= value <= ceiling:
        raise ValueError(f'{name} must be between 1 and {ceiling}')
    return value


def model_endpoint(base):
    try:
        u = urlsplit(base)
        allowed = {'api.deepseek.com'} | {
            h.strip().lower() for h in os.environ.get('DEEPSEEK_ALLOWED_HOSTS', '').split(',') if h.strip()
        }
        valid = (u.scheme == 'https' and u.hostname in allowed and
                 not u.username and not u.password and not u.query and not u.fragment and
                 u.port in (None, 443))
    except ValueError:
        valid = False
    if not valid:
        raise FatalServiceError('模型地址必须是获准主机的 HTTPS 地址（无凭据、查询串或片段）')
    return base.rstrip('/') + '/chat/completions'


def paper_id(value):
    if not re.fullmatch(r'(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z.\-]*/\d{7})(?:v[1-9]\d*)?', str(value)):
        raise ValueError('Invalid arXiv ID')
    return str(value)


def category(value):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9-]*(?:\.[A-Za-z][A-Za-z0-9-]*)?', value):
        raise ValueError('Invalid arXiv category')
    return value


def bounded_body(response, limit=2_000_000, deadline=None):
    chunks, size = [], 0
    try:
        for chunk in response.iter_content(16384):
            if deadline is not None and time.monotonic() >= deadline:
                raise FatalServiceError('Service time budget exhausted')
            size += len(chunk)
            if size > limit:
                raise FatalServiceError('Service response exceeds size limit')
            chunks.append(chunk)
        return b''.join(chunks)
    finally:
        response.close()


def response_json(response, limit=2_000_000, deadline=None):
    try:
        return json.loads(bounded_body(response, limit, deadline))
    except (ValueError, UnicodeError):
        raise RuntimeError('Service returned invalid JSON') from None

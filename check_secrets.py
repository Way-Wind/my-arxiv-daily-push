"""Scan tracked text before commit/CI. Reports paths only, never matched values.

Heuristic detection is not a guarantee; review staged changes as well.
"""
import re
import subprocess
from pathlib import Path

PATTERNS = [
    re.compile(r'\bsk-[A-Za-z0-9]{20,}\b'),
    re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'["\x27]app_secret["\x27]\s*:\s*["\x27][a-fA-F0-9]{32}["\x27]'),
]


def scan():
    root = Path(__file__).resolve().parent
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root).decode().split('\0')
    bad = []
    for name in filter(None, names):
        if Path(name).name == 'config.json':
            bad.append(name)
            continue
        try:
            text = (root / name).read_text(encoding='utf-8')
        except (UnicodeError, FileNotFoundError):
            continue
        if any(p.search(text) for p in PATTERNS):
            bad.append(name)
    for name in bad:
        print(f'Possible credential in tracked file: {name}')
    return bool(bad)


if __name__ == '__main__':
    raise SystemExit(scan())

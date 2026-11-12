"""Commit only generated data; use a step-scoped token without persisting credentials."""
import base64
import os
import subprocess


def git(*args, env=None):
    return subprocess.run(['git', *args], check=True, env=env, capture_output=True)


def export_commit():
    commit = git('rev-parse', 'HEAD').stdout.decode().strip()
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as f:
            f.write('commit=' + commit + '\n')


def persist():
    git('config', 'user.name', 'GitHub Actions')
    git('config', 'user.email', 'actions@users.noreply.github.com')
    git('add', '-f', '--', 'pages', 'page-data', 'pending.json', 'last_pushed.json')
    diff = subprocess.run(['git', 'diff', '--cached', '--quiet'])
    if diff.returncode == 0:
        export_commit()
        return
    if diff.returncode != 1:
        raise RuntimeError('Unable to inspect generated data')
    git('commit', '-m', 'chore: persist digest and delivery checkpoint')
    token = os.environ['GH_TOKEN']
    auth = base64.b64encode(('x-access-token:' + token).encode()).decode()
    env = dict(os.environ, GIT_CONFIG_COUNT='1',
               GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
               GIT_CONFIG_VALUE_0='AUTHORIZATION: basic ' + auth)
    # No force, reset or automatic rebase: concurrent edits must fail visibly.
    try:
        git('push', 'origin', 'HEAD:main', env=env)
    except subprocess.CalledProcessError:
        raise RuntimeError('Checkpoint push failed; recover from workflow artifact before retrying') from None
    export_commit()


if __name__ == '__main__':
    persist()

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
import page_builder
import security
import summarizer
import wechat_push


class Response:
    def __init__(self, data, status=200):
        self.status_code = status
        self.body = json.dumps(data).encode()
        self.closed = False
    def iter_content(self, size):
        yield self.body
    def close(self):
        self.closed = True


class SecurityTests(unittest.TestCase):
    def pusher(self):
        return wechat_push.WeChatPusher('fake-app', 'fake-secret', 'fake-template', 'fake-user')

    def test_failed_auth_never_counts(self):
        with patch('wechat_push.requests.get', return_value=Response({'access_token':'fake-token', 'expires_in':7200})), patch('wechat_push.requests.post', return_value=Response({'errcode':40001})) as post:
            confirmed = []
            with self.assertRaises(RuntimeError):
                self.pusher().send_batch([{'data':{}}], on_success=confirmed.append)
            self.assertEqual(confirmed, [])
            self.assertEqual(post.call_count, 2)

    def test_refresh_then_success(self):
        with patch('wechat_push.requests.get', return_value=Response({'access_token':'fake-token', 'expires_in':7200})), patch('wechat_push.requests.post', side_effect=[Response({'errcode':42001}), Response({'errcode':0})]):
            confirmed = []
            self.assertEqual(self.pusher().send_batch([{'data':{}}], on_success=confirmed.append), 1)
            self.assertEqual(confirmed, [0])

    def test_partial_batch_progress(self):
        p = self.pusher()
        with patch.object(p, 'send_template', side_effect=[True, RuntimeError('failed')]), patch('wechat_push.time.sleep'):
            confirmed = []
            with self.assertRaises(RuntimeError):
                p.send_batch([{'data':{}}, {'data':{}}], on_success=confirmed.append)
            self.assertEqual(confirmed, [0])

    def test_expiry(self):
        with patch('wechat_push.requests.get', return_value=Response({'access_token':'fake-token', 'expires_in':100})) as get, patch('wechat_push.time.monotonic', return_value=100) as clock:
            p = self.pusher()
            p.get_access_token(); p.get_access_token()
            self.assertEqual(get.call_count, 1)
            clock.return_value = 191
            p.get_access_token()
            self.assertEqual(get.call_count, 2)

    def test_invalid_lifetime(self):
        with patch('wechat_push.requests.get', return_value=Response({'access_token':'fake-token', 'expires_in':0})):
            p = self.pusher()
            with self.assertRaises(RuntimeError): p.get_access_token()
            self.assertIsNone(p._token)

    def test_safe_errors(self):
        with patch('wechat_push.requests.get', side_effect=wechat_push.requests.RequestException('secret=fake-secret&access_token=fake-token')):
            try: self.pusher().get_access_token()
            except RuntimeError as exc:
                import traceback
                trace = ''.join(traceback.format_exception(exc))
                self.assertNotIn('fake-secret', trace)
                self.assertNotIn('fake-token', trace)

    def test_endpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            for url in ['http://api.deepseek.com', 'https://api.deepseek.com.evil', 'https://user:pass@api.deepseek.com', 'https://api.deepseek.com?q=secret', 'https://api.deepseek.com#x']:
                with self.assertRaises(security.FatalServiceError): security.model_endpoint(url)
            self.assertEqual(security.model_endpoint('https://api.deepseek.com/v1'), 'https://api.deepseek.com/v1/chat/completions')
        with patch.dict(os.environ, {'DEEPSEEK_ALLOWED_HOSTS':'compatible.example'}):
            self.assertTrue(security.model_endpoint('https://compatible.example/v1').startswith('https://compatible.example/'))

    def test_html(self):
        p = {'id':'2609.00001', 'title':'<script>alert(1)</script>', 'summary':'<img src=x>', 'primary':'math.DG', 'categories':['math.DG']}
        html = page_builder.build_daily_html([p], {}, ['math.DG'], '2026-11-13')
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('Content-Security-Policy', html)
        for pid in ['2609.00001', 'math/0301234v2', 'hep-th/9901001']:
            security.paper_id(pid)
        p['id'] = '2609.00001"><img src=x onerror=alert(1)>'
        with self.assertRaises(ValueError): page_builder.build_daily_html([p], {}, ['math.DG'], '2026-11-13')
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
            with self.assertRaises(ValueError): page_builder.write_daily_page([], {}, [], '../x', d)

    def test_auth_fails_fast(self):
        p = {'id':'2609.00001', 'title':'sample', 'summary':'sample'}
        with patch('summarizer.requests.post', return_value=Response({'error':'fake-secret'},401)) as post, contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(security.FatalServiceError): summarizer.summarize_papers([p], 'fake-key')
            self.assertEqual(post.call_count, 1)
            self.assertNotIn('fake-secret',out.getvalue())
            self.assertFalse(post.call_args.kwargs['allow_redirects'])

    def test_request_budget(self):
        with patch.dict(os.environ, {'AI_MAX_REQUESTS':'1'}), patch('summarizer.requests.post', return_value=Response({'choices':[{'message':{'content':'{}'}}]})) as post:
            summarizer._budget.set([1, summarizer.time.monotonic()+30])
            summarizer._call_chat('fake-key','https://api.deepseek.com','model','text')
            with self.assertRaises(security.FatalServiceError): summarizer._call_chat('fake-key','https://api.deepseek.com','model','text')
            self.assertEqual(post.call_count,1)

    def test_body_limit(self):
        r = Response({'value':'x'*100})
        with self.assertRaises(security.FatalServiceError): security.response_json(r,10)
        self.assertTrue(r.closed)

    def test_time_budget_stops_before_network(self):
        summarizer._budget.set([5, summarizer.time.monotonic() - 1])
        with patch('summarizer.requests.post') as post:
            with self.assertRaises(security.FatalServiceError):
                summarizer._call_chat('fake-key', 'https://api.deepseek.com', 'model', 'text')
            post.assert_not_called()

    def test_paper_cap(self):
        papers = [{'id': '2609.00001', 'title':'sample', 'summary':'sample'},
                  {'id': '2609.00002', 'title':'sample', 'summary':'sample'}]
        with patch.dict(os.environ, {'AI_MAX_PAPERS':'1'}), patch('summarizer._call_with_retry', return_value={'2609.00001':{'title_zh':'sample'}}) as call, contextlib.redirect_stdout(io.StringIO()):
            result = summarizer.summarize_papers(papers, 'fake-key')
            self.assertEqual(set(result), {'2609.00001'})
            self.assertEqual(call.call_args.kwargs['expected_ids'], ['2609.00001'])

    def test_offline_preview_never_calls_network(self):
        with patch.object(sys, 'argv', ['main.py', '--offline-preview']), patch('page_builder.write_daily_page') as write, patch('requests.get', side_effect=AssertionError('Network forbidden')), patch('requests.post', side_effect=AssertionError('Network forbidden')), patch('main.load_config', side_effect=AssertionError('Credentials forbidden')), contextlib.redirect_stdout(io.StringIO()):
            main.main()
            write.assert_called_once()

    def test_atomic_state(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d, patch.object(main,'STATE_FILE', str(Path(d)/'state.json')):
            main._save_pushed_ids({'2609.00001'})
            self.assertEqual(main._load_pushed_ids(), {'2609.00001'})
            self.assertFalse(Path(main.STATE_FILE+'.tmp').exists())


if __name__ == '__main__': unittest.main()

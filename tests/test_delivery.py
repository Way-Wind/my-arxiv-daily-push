import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import delivery
import main
import page_builder


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.cwd = Path.cwd()
        os.chdir(self.temp.name)
        Path('pages').mkdir()
        Path('pages/daily-2026-11-13.html').write_bytes(b'published')
        self.url = 'https://example.github.io/repo/daily-2026-11-13.html'
        self.config = dict(app_id='dummy', app_secret='dummy', template_id='dummy', user_openid='dummy')

    def tearDown(self):
        os.chdir(self.cwd)
        self.temp.cleanup()

    def prepare(self, two=False):
        papers = [{'id':'2609.00001'}] + ([{'id':'2609.00002'}] if two else [])
        messages = [{'data':{}, 'url':self.url} for _ in papers]
        delivery.prepare(messages, papers, 'detailed', self.url)

    def test_prepare_has_no_credentials(self):
        self.prepare()
        batch = delivery.pending()
        self.assertEqual(batch['sha256'], hashlib.sha256(b'published').hexdigest())
        self.assertNotIn('app_secret', Path('pending.json').read_text())

    def test_failed_page_never_sends(self):
        self.prepare()
        with patch('delivery.verify_page', side_effect=RuntimeError('unavailable')), patch('delivery.WeChatPusher') as sender:
            with self.assertRaises(RuntimeError): delivery.notify(self.config,'https://example.github.io/repo')
            sender.assert_not_called()
        self.assertTrue(delivery.pending())
        self.assertFalse(Path('last_pushed.json').exists())

    def test_partial_failure_and_resume(self):
        self.prepare(two=True)
        with patch('delivery.verify_page'), patch('delivery.time.sleep'), patch('delivery.WeChatPusher') as sender:
            sender.return_value.send_template.side_effect = [True, RuntimeError('failed')]
            with self.assertRaises(RuntimeError): delivery.notify(self.config,'https://example.github.io/repo')
            self.assertEqual(json.loads(Path('last_pushed.json').read_text())['ids'], ['2609.00001'])
            self.assertEqual(len(delivery.pending()['entries']),1)
            sender.return_value.send_template.side_effect = [True]
            delivery.notify(self.config,'https://example.github.io/repo')
            self.assertIsNone(delivery.pending())
            self.assertEqual(len(json.loads(Path('last_pushed.json').read_text())['ids']),2)

    def test_state_saved_before_outbox_ack_prevents_duplicate(self):
        self.prepare()
        delivery.save('last_pushed.json',{'ids':['2609.00001']})
        with patch('delivery.verify_page'), patch('delivery.WeChatPusher') as sender:
            delivery.notify(self.config,'https://example.github.io/repo')
            sender.return_value.send_template.assert_not_called()
            self.assertIsNone(delivery.pending())

    def test_prepare_reuses_existing_outbox_without_model(self):
        self.prepare()
        before = Path('pending.json').read_bytes()
        with patch.object(sys,'argv',['main.py','--prepare-only']), patch('main.load_config',side_effect=AssertionError('must not read model key')), contextlib.redirect_stdout(io.StringIO()):
            main.main()
        self.assertEqual(Path('pending.json').read_bytes(),before)

    def test_untrusted_page_url_rejected(self):
        self.prepare()
        batch = delivery.pending()
        with patch('delivery.requests.get') as get:
            with self.assertRaises(RuntimeError): delivery.verify_page(batch,'https://other.example/repo')
            get.assert_not_called()

    def test_page_hash_mismatch_blocks(self):
        self.prepare()
        class Response:
            status_code = 200
            def iter_content(self, size): yield b'old page'
            def close(self): pass
        with patch('delivery.requests.get',return_value=Response()) as get, patch('delivery.time.sleep'):
            with self.assertRaises(RuntimeError): delivery.verify_page(delivery.pending(),'https://example.github.io/repo')
            self.assertEqual(get.call_count,5)

    def test_same_day_merge(self):
        p = dict(id='2609.00001', title='First', summary='text', primary='math.DG', categories=['math.DG'])
        page_builder.write_daily_page([p],{},['math.DG'],'2026-11-13')
        p = dict(p,id='2609.00002',title='Second')
        page_builder.write_daily_page([p],{},['math.DG'],'2026-11-13')
        html = Path('pages/daily-2026-11-13.html').read_text(encoding='utf-8')
        self.assertIn('First',html)
        self.assertIn('Second',html)
        data = json.loads(Path('page-data/2026-11-13.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data['papers']),2)


if __name__ == '__main__': unittest.main()

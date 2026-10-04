"""Offline checks for graph wiring, model integrity and supervisor cleanup."""
import contextlib
import hashlib
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_workflows
import download_models
import runtime


class PipelineTests(unittest.TestCase):
    def test_workflows_are_regenerated_and_connected(self):
        for name, reference in [('qwen21_q8_text_to_image', False), ('qwen21_q8_reference_image', True)]:
            graph = build_workflows.build(reference)
            actual = json.loads((ROOT / 'workflows' / (name + '.json')).read_text())
            self.assertEqual(actual, graph.workflow(name))
            self.assertEqual(json.loads((ROOT / 'workflows/api' / (name + '.json')).read_text()), graph.api)
            nodes = {n['id']: n for n in actual['nodes']}
            for id, src, slot, dst, index, kind in actual['links']:
                self.assertIn(id, nodes[src]['outputs'][slot]['links'])
                self.assertEqual(id, nodes[dst]['inputs'][index]['link'])
                self.assertEqual(kind, nodes[dst]['inputs'][index]['type'])
            # Reference must use both vision and VAE conditioning plus the matched latent size.
            if reference:
                self.assertEqual(graph.api['4']['inputs']['images.image_1'], ['6', 0])
                self.assertEqual(graph.api['4']['inputs']['vae'], ['3', 0])
                self.assertEqual(graph.api['7']['inputs']['latent_image'], ['4', 2])
            else:
                self.assertNotIn('images.image_1', graph.api['4']['inputs'])

    def test_hash_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model'
            data = b'valid model contents'
            entry = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            path.write_bytes(data)
            self.assertTrue(download_models.verified(path, entry))
            path.write_bytes(b'x' * len(data))
            self.assertFalse(download_models.verified(path, entry))

    def test_missing_runtime_node_is_actionable(self):
        with self.assertRaisesRegex(RuntimeError, 'Missing node UnetLoaderGGUF'):
            runtime.check_nodes({})

    def test_tunnel_failure_stops_comfy_and_removes_url(self):
        # Real local child processes, with no GPU, download or external network access.
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            state = base / '.runtime'
            comfy = state / 'ComfyUI'
            comfy.mkdir(parents=True)
            (state / 'bin').mkdir()
            shutil.copytree(ROOT / 'workflows', base / 'workflows')
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            schema = {n['class_type']: {'input': {}} for n in build_workflows.build().api.values()}
            schema.update({n['class_type']: {'input': {}} for n in build_workflows.build(True).api.values()})
            (comfy / 'main.py').write_text(
                'import os,json\nfrom pathlib import Path\nfrom http.server import BaseHTTPRequestHandler,HTTPServer\n'
                f'Path({str(base / "comfy.pid")!r}).write_text(str(os.getpid()))\n'
                'class Handler(BaseHTTPRequestHandler):\n'
                ' def do_GET(self):\n'
                f'  data=json.dumps({schema!r}).encode();self.send_response(200);self.end_headers();self.wfile.write(data)\n'
                f'HTTPServer(("127.0.0.1",{port}),Handler).serve_forever()\n')
            tunnel = state / 'bin/cloudflared'
            tunnel.write_text(f'#!{sys.executable}\nimport time\nprint("https://test-public.trycloudflare.com",flush=True)\ntime.sleep(2)\nraise SystemExit(7)\n')
            tunnel.chmod(0o755)
            args = type('Args', (), dict(check=False, local=False, direct=False, port=port,
                                        novram=False, cpu_vae=False, startup_timeout=10, tunnel_timeout=10))()
            with patch.multiple(runtime, ROOT=base, STATE=state, COMFY=comfy), \
                 patch.object(runtime, 'check_install'), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'Tunnel exited'):
                    runtime.supervise(args)
            pid = int((base / 'comfy.pid').read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            self.assertFalse((state / 'public-url.txt').exists())


if __name__ == '__main__':
    unittest.main()

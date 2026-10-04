"""Run a small real GPU job and verify that ComfyUI produced an image."""
import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def request(url, data=None, content_type='application/json'):
    req = urllib.request.Request(url, data=data, headers={'Content-Type': content_type})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8188')
    parser.add_argument('--reference', type=Path, help='Upload a reference and test editing too')
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    url = args.url.rstrip('/')
    name = 'qwen21_q8_reference_image' if args.reference else 'qwen21_q8_text_to_image'
    graph = json.loads((ROOT / 'workflows/api' / (name + '.json')).read_text())
    graph['7']['inputs']['steps'] = 4
    graph['9']['inputs']['filename_prefix'] = 'smoke/qwen21'
    if args.reference:
        boundary = uuid.uuid4().hex
        # Use an ASCII name independent of the supplied filename.
        filename = 'smoke-' + uuid.uuid4().hex + args.reference.suffix.lower()
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
                'Content-Type: application/octet-stream\r\n\r\n').encode()
        body += args.reference.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
        upload = request(url + '/upload/image', body, f'multipart/form-data; boundary={boundary}')
        graph['6']['inputs']['image'] = '/'.join(filter(None, [upload.get('subfolder'), upload['name']]))
        graph['4']['inputs']['resolution'] = 512
    else:
        graph['6']['inputs'].update(width=512, height=512)
    result = request(url + '/prompt', json.dumps({'prompt': graph, 'client_id': uuid.uuid4().hex}).encode())
    if result.get('error') or result.get('node_errors') or not result.get('prompt_id'):
        raise SystemExit(f'Workflow rejected: {result}')
    prompt_id = result['prompt_id']
    print(f'Queued {prompt_id}. This checks execution, not final image quality.', flush=True)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        history = request(url + '/history/' + prompt_id).get(prompt_id)
        if history:
            if history.get('status', {}).get('status_str') == 'error':
                raise SystemExit(f'Generation failed: {history["status"]}')
            images = history.get('outputs', {}).get('9', {}).get('images', [])
            if images:
                print('PASS: image saved:', images)
                return
            raise SystemExit(f'Job ended without a saved image: {history.get("status")}')
        time.sleep(2)
    raise SystemExit(f'Timed out waiting for {prompt_id}; the job may still be queued/running. Check ComfyUI.')


if __name__ == '__main__':
    main()

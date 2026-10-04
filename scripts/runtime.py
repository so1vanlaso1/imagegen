"""Supervise ComfyUI and an anonymous HTTPS tunnel as one foreground process."""
import argparse
import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.runtime'
COMFY = STATE / 'ComfyUI'


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def check_install():
    if not (COMFY / 'main.py').is_file():
        raise RuntimeError('ComfyUI missing. Run ./setup.sh first.')
    manifest = json.loads((ROOT / 'config/models.json').read_text())
    for entry in manifest['files']:
        path = COMFY / 'models' / entry['destination']
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise RuntimeError(f'Missing/incomplete {path.name}. Run ./setup.sh to download and verify models.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check nvidia-smi, NVIDIA driver and GPU access on Vast.ai.')
    print(f'GPU: {torch.cuda.get_device_name(0)}; VRAM: '
          f'{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB', flush=True)


def check_nodes(info):
    for file in (ROOT / 'workflows/api').glob('*.json'):
        for node in json.loads(file.read_text()).values():
            name = node['class_type']
            if name not in info:
                raise RuntimeError(f'Missing node {name}. Rerun setup with the pinned ComfyUI/GGUF revisions.')
            # Confirm actual runtime model names, devices and sampler choices.
            inputs = info[name].get('input', {})
            fields = {**inputs.get('required', {}), **inputs.get('optional', {})}
            for key, value in node['inputs'].items():
                field = fields.get(key)
                if field and isinstance(field[0], list) and not isinstance(value, list) and value not in field[0]:
                    raise RuntimeError(f'{name}.{key}: {value!r} is unavailable in this installation.')


def stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def pump(process, logfile, callback=None):
    def read():
        with logfile.open('w', buffering=1) as log:
            for line in process.stdout:
                log.write(line)
                print(line, end='', flush=True)
                if callback:
                    callback(line)
    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    return thread


def supervise(args):
    STATE.mkdir(exist_ok=True)
    (STATE / 'logs').mkdir(exist_ok=True)
    # flock automatically releases on crash; there are no stale PID files to trust.
    with (STATE / 'run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This checkout is already running.') from None
        check_install()
        if args.check:
            print('Installation and model sizes look good. setup.sh verifies full SHA256 hashes.')
            return
        public_file = STATE / 'public-url.txt'
        public_file.unlink(missing_ok=True)
        if not args.local and not args.direct and not (STATE / 'bin/cloudflared').is_file():
            raise RuntimeError('cloudflared missing. Rerun ./setup.sh.')
        with socket.socket() as probe:
            try:
                probe.bind(('0.0.0.0' if args.direct else '127.0.0.1', args.port))
            except OSError:
                raise RuntimeError(f'Port {args.port} is already in use. Choose --port NUMBER.') from None
        user = STATE / 'user'
        (user / 'default/workflows').mkdir(parents=True, exist_ok=True)
        for workflow in (ROOT / 'workflows').glob('*.json'):
            target = user / 'default/workflows' / workflow.name
            if not target.exists():
                shutil.copyfile(workflow, target)
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        cmd = [sys.executable, str(COMFY / 'main.py'), '--listen',
               '0.0.0.0' if args.direct else '127.0.0.1', '--port', str(args.port),
               '--user-directory', str(user), '--disable-auto-launch',
               '--disable-dynamic-vram', '--novram' if args.novram else '--lowvram',
               '--disable-comfy-compiler', '--reserve-vram', '1.0',
               '--use-pytorch-cross-attention', '--preview-method', 'none']
        if args.cpu_vae:
            cmd.append('--cpu-vae')
        comfy = tunnel = None
        threads = []
        try:
            comfy = subprocess.Popen(cmd, cwd=COMFY, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            threads.append(pump(comfy, STATE / 'logs/comfyui.log'))
            origin = f'http://127.0.0.1:{args.port}'
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if comfy.poll() is not None:
                    raise RuntimeError(f'ComfyUI exited ({comfy.returncode}); see .runtime/logs/comfyui.log.')
                try:
                    info = get_json(origin + '/object_info')
                    break
                except (OSError, ValueError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError('ComfyUI startup timed out; see .runtime/logs/comfyui.log.') from None
                    time.sleep(1)
            check_nodes(info)
            print(f'ComfyUI ready: {origin}', flush=True)
            if args.direct:
                print(f'Public direct mode: open the Vast.ai mapped TCP port for {args.port}. No authentication.', flush=True)
            elif not args.local:
                def tunnel_line(line):
                    match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
                    if match:
                        public_file.write_text(match.group(0) + '\n')
                        print(f'\nPUBLIC COMFYUI URL (no login): {match.group(0)}\n', flush=True)
                # Explicit empty config prevents an unrelated ~/.cloudflared config from interfering.
                command = [str(STATE / 'bin/cloudflared'), '--no-autoupdate', 'tunnel',
                           '--config', '/dev/null', '--url', origin, '--protocol', 'http2']
                tunnel = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True, bufsize=1)
                threads.append(pump(tunnel, STATE / 'logs/tunnel.log', tunnel_line))
                deadline = time.monotonic() + args.tunnel_timeout
                while not public_file.exists():
                    if comfy.poll() is not None or tunnel.poll() is not None:
                        raise RuntimeError('ComfyUI or tunnel exited during startup; see .runtime/logs/.')
                    if time.monotonic() >= deadline:
                        raise RuntimeError('No public URL received; inspect tunnel.log or use --direct with a Vast port mapping.')
                    time.sleep(0.5)
            print('Load a workflow in the Workflows sidebar, then Run. Ctrl+C stops the server and tunnel.', flush=True)
            while True:
                if comfy.poll() is not None:
                    raise RuntimeError(f'ComfyUI exited ({comfy.returncode}); see comfyui.log.')
                if tunnel is not None and tunnel.poll() is not None:
                    raise RuntimeError(f'Tunnel exited ({tunnel.returncode}); see tunnel.log. Rerun ./run.sh for a new URL.')
                time.sleep(0.5)
        finally:
            stop_process(tunnel)
            stop_process(comfy)
            for thread in threads:
                thread.join(timeout=2)
            for process in (tunnel, comfy):
                if process is not None and process.stdout is not None:
                    process.stdout.close()
            public_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--local', action='store_true', help='Localhost only; no tunnel')
    mode.add_argument('--direct', action='store_true', help='Listen on 0.0.0.0; use a Vast.ai mapped port')
    parser.add_argument('--port', type=int, default=8188)
    parser.add_argument('--cpu-vae', action='store_true', help='Slower VAE on CPU if encode/decode runs out of VRAM')
    parser.add_argument('--novram', action='store_true', help='More aggressive diffusion offload, retaining Q8')
    parser.add_argument('--check', action='store_true', help='Check installed files and CUDA, then exit')
    parser.add_argument('--startup-timeout', type=float, default=600)
    parser.add_argument('--tunnel-timeout', type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.startup_timeout <= 0 or args.tunnel_timeout <= 0:
        parser.error('Choose a valid port and positive timeouts.')
    import signal
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        supervise(args)
    except KeyboardInterrupt:
        print('\nStopped.', flush=True)
    except (RuntimeError, OSError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()

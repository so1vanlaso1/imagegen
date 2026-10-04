"""Download exactly the pinned Q8 model and companions, verifying upstream hashes."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'config/models.json'
MODEL_DIR = ROOT / '.runtime/ComfyUI/models'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path, entry):
    return path.is_file() and path.stat().st_size == entry['size'] and sha256(path) == entry['sha256']


def main():
    from huggingface_hub import hf_hub_download

    manifest = json.loads(MANIFEST.read_text())
    missing = []
    for entry in manifest['files']:
        target = MODEL_DIR / entry['destination']
        if verified(target, entry):
            print(f'Already verified: {target.name}', flush=True)
        else:
            if target.exists() or target.is_symlink():
                raise SystemExit(f'Unexpected/corrupt model at {target}; move it aside and rerun setup.')
            missing.append(entry)
    needed = sum(entry['size'] for entry in missing)
    # Resumed cache downloads may already occupy some of this space.
    if shutil.disk_usage(ROOT).free < needed + 2 * 2**30:
        print(f'Full missing downloads total {needed / 2**30:.1f} GiB; ensure space remains for the cache.', flush=True)
    for entry in missing:
        print(f'Downloading {entry["source"]} ({entry["size"] / 1e9:.2f} GB). Interrupts resume on rerun.', flush=True)
        cached = Path(hf_hub_download(repo_id=manifest['repo_id'], revision=manifest['revision'],
                                     filename=entry['source'], cache_dir=ROOT / '.runtime/hf-cache'))
        print(f'Checking SHA256: {cached.name}', flush=True)
        if not verified(cached, entry):
            raise SystemExit(f'Checksum mismatch at {cached}. Remove the corrupt cached file and rerun setup.')
        target = MODEL_DIR / entry['destination']
        target.parent.mkdir(parents=True, exist_ok=True)
        # One copy on disk. Relative symlinks survive relocating the whole repo.
        import os
        target.symlink_to(os.path.relpath(cached, target.parent))
    print('All model files verified.', flush=True)


if __name__ == '__main__':
    main()


"""Lightweight Linux resource check before installing or downloading anything."""
import os
import shutil
from pathlib import Path


def main():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0]) * 1024
    # Vast containers can have less memory assigned than the host.
    limits = [values['MemTotal']]
    for file in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            limit = Path(file).read_text().strip()
            if limit.isdigit():
                limits.append(int(limit))
        except OSError:
            pass
    ram = min(limits) / 2**30
    free = shutil.disk_usage(Path(__file__).resolve().parents[1]).free / 2**30
    print(f'System/container RAM: {ram:.1f} GiB. Free disk: {free:.1f} GiB.')
    if ram < 24 and os.environ.get('ALLOW_LOW_RAM') != '1':
        raise SystemExit('Q8 plus the text encoder needs substantial CPU RAM. Rent >=32 GB RAM '
                         '(24 GiB minimum check), or explicitly set ALLOW_LOW_RAM=1 to try slower swap/offload.')
    if free < 40:
        print('Disk space is low for a fresh install; provision 60+ GB total. Existing installs need less.')


if __name__ == '__main__':
    main()

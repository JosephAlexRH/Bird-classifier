"""Verified live copy of bird_images to a specific mounted USB volume.

This is an incremental live backup, not a consistent end-of-deployment archive.
No source files are changed; old destination versions and deleted files remain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def check_mount(mount, uuid):
    # --mountpoint requires this exact path, not its parent/root filesystem.
    actual = subprocess.check_output(
        ['findmnt', '--noheadings', '--raw', '--mountpoint', str(mount),
         '--output', 'UUID'], text=True).strip()
    if actual.casefold() != uuid.casefold():
        raise RuntimeError(f'Expected USB UUID {uuid}; found {actual!r}')


def copy_stable(source, target, history):
    """Stage, hash, recheck source, preserve prior version, then replace."""
    if source.is_symlink() or target.is_symlink():
        raise ValueError(f'Refusing symlink: {source} or {target}')
    before = source.stat()
    source_hash = digest(source)
    if target.exists() and digest(target) == source_hash:
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Source changed during verification: {source}')
        return source_hash, False
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.copy-', dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output, source.open('rb') as incoming:
            shutil.copyfileobj(incoming, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        copied_hash = digest(temporary)
        after = source.stat()
        if (copied_hash != source_hash or digest(source) != source_hash or
                (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
            raise RuntimeError(f'Source changed during copy: {source}')
        if target.exists():
            old_hash = digest(target)
            previous = history / old_hash
            previous.parent.mkdir(parents=True, exist_ok=True)
            if not previous.exists():
                shutil.copyfile(target, previous)
            if digest(previous) != old_hash:
                raise RuntimeError(f'Prior version verification failed: {previous}')
        os.replace(temporary, target)
        if digest(target) != source_hash:
            raise RuntimeError(f'Destination verification failed: {target}')
        return source_hash, True
    finally:
        temporary.unlink(missing_ok=True)


def backup(source, destination):
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate trees')
    destination.mkdir(parents=True, exist_ok=True)
    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'kind': 'incremental_live_backup', 'consistent_snapshot': False,
              'source': str(source), 'files': [], 'errors': [], 'copied': 0}
    for root, directories, names in os.walk(
            source, followlinks=False,
            onerror=lambda error: report['errors'].append(str(error))):
        for directory in list(directories):
            if (Path(root) / directory).is_symlink():
                report['errors'].append(f'Refusing symlink directory: {Path(root) / directory}')
                directories.remove(directory)
        for name in sorted(names):
            path = Path(root) / name
            relative = path.relative_to(source)
            try:
                sha, copied = copy_stable(path, destination / 'bird_images' / relative,
                                          destination / 'previous_versions')
                report['files'].append({'path': relative.as_posix(), 'sha256': sha})
                report['copied'] += int(copied)
            except Exception as error:
                report['errors'].append(f'{relative}: {error}')
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['verified'] = not report['errors']
    # Each report describes only files observed in this pass. New files wait for
    # the next run; changing logs may fail verification and are retried then.
    reports = destination / 'reports'
    reports.mkdir(exist_ok=True)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '.json'
    with (reports / name).open('w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--mount', type=Path, required=True)
    parser.add_argument('--uuid', required=True)
    args = parser.parse_args()
    mount = args.mount.resolve()
    check_mount(mount, args.uuid)
    if args.check_only:
        return 0
    if args.source is None:
        parser.error('--source is required unless --check-only is used')
    # Prevent overlapping manual/timer runs on Linux.
    import fcntl
    with (mount / '.birdhub-backup.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = backup(args.source.resolve(), mount / 'birdhub-backup')
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}, indent=2))
    return 0 if result['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

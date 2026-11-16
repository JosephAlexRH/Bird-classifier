"""Erase archived Pi captures only after verifying their source inventories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timezone


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_tree(root, inventory):
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'Expected an actual directory: {root}')
    expected = {}
    for item in inventory['files']:
        relative = item['path']
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or path == root.resolve() or relative in expected:
            raise ValueError(f'Unsafe inventory path: {relative}')
        expected[relative] = item
    actual = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Refusing symbolic link: {path}')
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = path
    if set(actual) != set(expected):
        raise ValueError(f'Files changed since archive: {root}. Archive again before erasing.')
    for relative, path in actual.items():
        item = expected[relative]
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError(f'File changed since archive: {path}')
    return {item['sha256'] for item in expected.values()}


def check_previous(previous, by_source, hashes):
    if previous.is_symlink():
        raise ValueError('Previous versions path is a symlink')
    if str(previous) in by_source:
        check_tree(previous, by_source[str(previous)])
    else:
        for path in previous.rglob('*'):
            if path.is_symlink() or (path.is_file() and sha256(path) not in hashes):
                raise ValueError(f'Unarchived earlier version: {path}. Preserve it before resetting.')


def checked_targets(home, mount, inventories):
    active = mount / 'birdhub-backup/bird_images'
    if not (home / 'bird_images').is_symlink() or (home / 'bird_images').resolve() != active:
        raise ValueError('Unexpected capture path; refusing reset')
    targets = [active, *sorted(home.glob('bird_images.before-usb-*')),
               *sorted((mount / 'birdhub-backup').glob('bird_images.completed-*'))]
    by_source = {item['source_root']: item for item in inventories}
    hashes = set()
    for target in targets:
        # Reject paths outside the two explicitly allowed parents before delete.
        if target.is_symlink() or target.resolve() != target or target.parent not in {home, mount / 'birdhub-backup'}:
            raise ValueError(f'Unsafe deletion target: {target}')
        if str(target) not in by_source:
            raise ValueError(f'No verified inventory supplied for {target}')
        hashes.update(check_tree(target, by_source[str(target)]))
    previous = mount / 'birdhub-backup/previous_versions'
    if previous.exists():
        check_previous(previous, by_source, hashes)
        targets.append(previous)
    return active, targets


def main():
    import pwd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, action='append', required=True)
    parser.add_argument('--deployment-id', required=True)
    args = parser.parse_args()
    if os.geteuid() != 0 or not re.fullmatch(r'[A-Za-z0-9_-]+', args.deployment_id):
        raise ValueError('Run with sudo and supply a valid deployment ID')
    user = pwd.getpwnam('jrussell')
    home = Path(user.pw_dir).resolve()
    mount = Path('/mnt/bird-backup')
    uuid = subprocess.check_output(['findmnt', '-n', '-r', '-M', str(mount), '-o', 'UUID'], text=True).strip()
    if uuid.upper() != 'FC1B-D5B3':
        raise ValueError('Correct USB drive is not mounted')
    inventories = [json.loads(path.read_text()) for path in args.inventory]
    override = Path('/etc/systemd/system/birdhub.service.d/collection.conf')
    old_config = override.read_text()
    if len(re.findall(r'--deployment-id\s+\S+', old_config)) != 1:
        raise ValueError('Unexpected service command; no reset performed')
    config = re.sub(r'--deployment-id\s+\S+', '--deployment-id ' + args.deployment_id, old_config)
    subprocess.run(['systemctl', 'stop', 'birdhub.service'], check=True)
    # Every target is checked before the first deletion. Any mismatch leaves
    # all data intact and capture stopped for another archive.
    active, targets = checked_targets(home, mount, inventories)
    settings = {name: (active / name).read_bytes() for name in
                ('cameras.json', 'fleet.json', 'hub_settings.json') if (active / name).is_file()}
    audit = home / 'birdserver/deployment_exports'
    audit.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (audit / f'clean-reset-{stamp}.json').write_text(json.dumps({
        'deployment_id': args.deployment_id, 'checked_targets': list(map(str, targets)),
        'inventories': list(map(str, args.inventory)), 'status': 'verified_before_erasure'}, indent=2))
    override.with_name(override.name + '.before-clean-' + stamp).write_text(old_config)
    for target in targets:
        shutil.rmtree(target)
    active.mkdir()
    os.chown(active, user.pw_uid, user.pw_gid)
    for name, content in settings.items():
        path = active / name
        path.write_bytes(content)
        os.chown(path, user.pw_uid, user.pw_gid)
    override.write_text(config)
    os.sync()
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', 'start', 'birdhub.service'], check=True)
    print(f'Fresh deployment started: {args.deployment_id}. Archived captures erased from Pi SD and USB.')
    print('Camera settings, models, laptop archives, and archival audit records preserved.')


if __name__ == '__main__':
    main()

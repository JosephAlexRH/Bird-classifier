"""Read-only Pi inventories, verified deployment copies, and legacy HTTP rescue.

Raw acquisition files are never edited or deleted. An HTTP rescue is explicitly
not a complete raw archive when the old API cannot supply acquisition sidecars.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def build_inventory(root, excluded=()):
    root = Path(root).resolve(strict=True)
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Refusing a symbolic link in the raw archive: {path}')
        if not path.is_file():
            continue
        if path.relative_to(root).as_posix() in excluded:
            continue
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Source changed while hashing: {path}. Stop capture before inventory.')
        files.append({'path': path.relative_to(root).as_posix(), 'bytes': after.st_size,
                      'sha256': digest, 'mtime_ns': after.st_mtime_ns})
    return {'schema_version': 1, 'created_at': utc_now(), 'source_root': str(root),
            'files': files, 'file_count': len(files), 'byte_count': sum(f['bytes'] for f in files),
            'excluded_paths': list(excluded)}


def verify(root, inventory):
    root = Path(root).resolve(strict=True)
    errors, expected = [], set()
    # Windows merges directory names such as Models/ and models/. Preserve
    # original inventory spelling, but compare filesystem identities correctly.
    case_insensitive = os.path.normcase('A') == os.path.normcase('a')
    path_key = (lambda value: value.casefold()) if case_insensitive else (lambda value: value)
    entries = inventory.get('files')
    if not isinstance(entries, list) or len(entries) != inventory.get('file_count'):
        raise ValueError('Invalid source inventory.')
    for item in entries:
        relative = item['path']
        path = (root / relative).resolve()
        key = path_key(relative)
        if not path.is_relative_to(root) or path == root or key in expected:
            raise ValueError(f'Unsafe or duplicate inventory path: {relative}')
        expected.add(key)
        if not path.is_file():
            errors.append({'path': relative, 'error': 'missing'})
        elif path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            errors.append({'path': relative, 'error': 'size_or_sha256_mismatch'})
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    excluded = {path_key(p) for p in inventory.get('excluded_paths', [])}
    errors.extend({'path': p, 'error': 'not_in_source_inventory'} for p in sorted(actual)
                  if path_key(p) not in expected and path_key(p) not in excluded)
    return {'schema_version': 1, 'verified_at': utc_now(), 'root': str(root),
            'verified': not errors, 'file_count': len(entries),
            'byte_count': sum(f['bytes'] for f in entries), 'errors': errors}


def inventory_source(source, output, project=None):
    source, output = Path(source).expanduser().resolve(strict=True), Path(output).expanduser().resolve()
    if output.is_relative_to(source):
        raise ValueError('Write the inventory outside the capture directory.')
    if output.exists():
        raise FileExistsError(f'Use a new archive output directory: {output}')
    output.mkdir(parents=True)
    inventory = build_inventory(source)
    write_json(output / 'source_inventory.json', inventory)
    context = {'schema_version': 1, 'created_at': utc_now(), 'hostname': platform.node(),
               'platform': platform.platform(), 'machine': platform.machine(),
               'python': sys.version, 'source_inventory_sha256': sha256(output / 'source_inventory.json'),
               'packages': sorted(({'name': d.metadata['Name'], 'version': d.version}
                                   for d in importlib.metadata.distributions()), key=lambda d: str(d['name'])),
               'raw_capture_count': len(list(source.glob('capture_*.jpg'))),
               'snapshot_policy': 'cumulative_source_directory_no_source_deletion',
               'archive_tool_sha256': sha256(Path(__file__)),
               'assets': []}
    acquisition_count, deployments, cameras, received = 0, set(), set(), []
    for path in source.glob('capture_*.json'):
        if not re.fullmatch(r'capture_[A-Za-z0-9_]+', path.stem):
            continue
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        acquisition_count += 1
        deployments.add(record.get('deployment_id') or 'legacy_unspecified')
        camera_id = record.get('camera_id') or (record.get('device') or {}).get('camera_id')
        cameras.add(camera_id or 'legacy_unspecified')
        if isinstance(record.get('received_at'), str):
            received.append(record['received_at'])
    context.update(acquisition_sidecars=acquisition_count, deployment_ids=sorted(deployments),
                   camera_ids=sorted(cameras), first_received_at=min(received) if received else None,
                   last_received_at=max(received) if received else None)
    if project:
        project = Path(project).expanduser().resolve(strict=True)
        paths = list((project / 'server').glob('*.py')) + list((project / 'deploy').glob('*.sh'))
        for folder in ('Models', 'models'):
            base = project / folder
            if base.is_dir():
                paths.extend(p for p in base.rglob('*') if p.suffix in {'.tflite', '.json', '.txt', '.whl'})
        for path in sorted(set(paths)):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(project)
            destination = output / 'deployed_assets' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            context['assets'].append({'path': relative.as_posix(), 'sha256': sha256(destination)})
    write_json(output / 'deployment_context.json', context)
    write_json(output / 'audit_inventory.json', build_inventory(output, ('audit_inventory.json',)))
    print(json.dumps({'source_files': inventory['file_count'], 'source_bytes': inventory['byte_count'],
                      'captures': context['raw_capture_count'], 'inventory': str(output / 'source_inventory.json')}))


def rescue_http(pi_url, output):
    parsed = urlsplit(pi_url.strip())
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.path not in {'', '/'} or parsed.query or parsed.fragment:
        raise ValueError('Use a Pi HTTP(S) base URL without credentials or a path.')
    base = urlunsplit((parsed.scheme, parsed.netloc, '', '', ''))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    api_dir, captures_dir = output / '_api', output / 'captures'
    api_dir.mkdir(exist_ok=True)
    captures_dir.mkdir(exist_ok=True)

    def request(endpoint, limit=12 * 1024 * 1024):
        for attempt in range(3):
            try:
                with urlopen(base + endpoint, timeout=30) as response:
                    body = response.read(limit + 1)
                if len(body) > limit:
                    raise ValueError(f'Response too large: {endpoint}')
                return body
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1)

    catalog_path = api_dir / 'catalog.json'
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
        if catalog['pi_url'] != base:
            raise ValueError('This rescue snapshot belongs to a different Pi URL.')
    else:
        health = json.loads(request('/health'))
        cutoff = Path(health.get('latest_image') or '').stem
        if not re.fullmatch(r'capture_[A-Za-z0-9_]+', cutoff):
            raise ValueError('Pi did not report a usable latest capture.')
        records, after = [], ''
        while True:
            listing = json.loads(request('/api/captures?limit=100&after=' + after))
            batch = listing.get('captures', [])
            for capture in batch:
                cid = capture.get('capture_id', '')
                if not re.fullmatch(r'capture_[A-Za-z0-9_]+', cid) or cid <= after:
                    raise ValueError('Invalid or non-increasing capture ID from Pi.')
                if cid <= cutoff:
                    records.append(capture)
                after = cid
            if not batch or after >= cutoff or not listing.get('has_more'):
                break
        catalog = {'schema_version': 1, 'pi_url': base, 'created_at': utc_now(),
                   'cutoff_capture_id': cutoff, 'captures': records}
        write_json(api_dir / 'health_at_snapshot.json', health)
        write_json(catalog_path, catalog)
    missing_metadata = []
    for capture in catalog['captures']:
        cid, expected = capture['capture_id'], capture.get('sha256')
        if not re.fullmatch(r'[a-f0-9]{64}', expected or ''):
            raise ValueError(f'No source SHA-256 for {cid}; cannot claim verified rescue.')
        path = captures_dir / f'{cid}.jpg'
        if path.exists():
            if sha256(path) != expected:
                raise ValueError(f'Existing local image differs: {path}. It has not been overwritten.')
        else:
            image = request(f'/api/captures/{cid}/image')
            if hashlib.sha256(image).hexdigest() != expected:
                raise ValueError(f'Pi image checksum mismatch for {cid}')
            write_bytes(path, image)
        if capture.get('metadata_url'):
            raw = request(f'/api/captures/{cid}/metadata')
            metadata = json.loads(raw)
            if metadata.get('sha256') != expected:
                raise ValueError(f'Acquisition sidecar does not match {cid}')
            sidecar = captures_dir / f'{cid}.json'
            if sidecar.exists() and sidecar.read_bytes() != raw:
                raise ValueError(f'Existing acquisition sidecar differs: {sidecar}')
            write_bytes(sidecar, raw)
        else:
            missing_metadata.append(cid)
        # These are explicitly API exports, not claims of byte-identical source files.
        for key in ('inference', 'motion', 'review'):
            if isinstance(capture.get(key), dict):
                write_json(api_dir / f'{cid}.{key}.json', capture[key])
    report = {'schema_version': 1, 'created_at': utc_now(), 'kind': 'verified_image_rescue',
              'pi_url': base, 'cutoff_capture_id': catalog['cutoff_capture_id'],
              'images_verified': len(catalog['captures']), 'source_images_deleted': False,
              'complete_raw_archive': False,
              'missing_original_acquisition_sidecars': missing_metadata,
              'limitation': 'API rescue is not a complete filesystem archive. Preserve the Pi and use the SSH archive for exact sidecars, crops, logs, profiles and manifests.'}
    write_json(output / 'rescue_report.json', report)
    inventory = build_inventory(captures_dir)
    write_json(output / 'image_inventory.json', inventory)
    verification = verify(captures_dir, inventory)
    write_json(output / 'image_verification.json', verification)
    print(json.dumps({'images_verified_against_pi_sha256': len(catalog['captures']),
                      'missing_acquisition_sidecars': len(missing_metadata),
                      'complete_raw_archive': False, 'snapshot': str(output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    source = sub.add_parser('inventory')
    source.add_argument('--source', required=True, type=Path)
    source.add_argument('--output', required=True, type=Path)
    source.add_argument('--project', type=Path)
    check = sub.add_parser('verify')
    check.add_argument('--root', required=True, type=Path)
    check.add_argument('--inventory', required=True, type=Path)
    check.add_argument('--report', required=True, type=Path)
    rescue = sub.add_parser('rescue')
    rescue.add_argument('--pi-url', required=True)
    rescue.add_argument('--output', required=True, type=Path)
    seal = sub.add_parser('seal')
    seal.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'inventory':
        inventory_source(args.source, args.output, args.project)
    elif args.command == 'verify':
        report = verify(args.root, json.loads(args.inventory.read_text(encoding='utf-8-sig')))
        write_json(args.report, report)
        print(json.dumps({'verified': report['verified'], 'files': report['file_count'],
                          'bytes': report['byte_count'], 'errors': report['errors'][:20]}, indent=2))
        if not report['verified']:
            raise SystemExit(1)
    elif args.command == 'rescue':
        rescue_http(args.pi_url, args.output)
    else:
        root = args.root.resolve(strict=True)
        inventory = build_inventory(root, ('_archive_inventory.json',))
        write_json(root / '_archive_inventory.json', inventory)
        print(json.dumps({'sealed': str(root), 'files': inventory['file_count']}))


if __name__ == '__main__':
    main()

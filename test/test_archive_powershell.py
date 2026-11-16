"""Exercise the Windows archive workflow with fake SSH/SCP and real hashing."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.archive_pi_deployment import build_inventory, write_json


@unittest.skipUnless(shutil.which('powershell'), 'Windows PowerShell required')
class ArchivePowerShellTests(unittest.TestCase):
    def run_workflow(self, source_root, retry=False, resume=False, legacy=False, verify_only=False, history=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'raw'
            audit = root / 'audit'
            raw.mkdir()
            audit.mkdir()
            (raw / 'capture_a.jpg').write_bytes(b'original image')
            (raw / 'capture_a.json').write_bytes(b'{"original": true}\n')
            (raw / 'crops').mkdir()
            (raw / 'crops/a.png').write_bytes(b'crop')
            (raw / '.hidden').write_bytes(b'hidden')
            inventory = build_inventory(raw)
            inventory['source_root'] = source_root
            write_json(audit / 'source_inventory.json', inventory)
            write_json(audit / 'audit_inventory.json', build_inventory(audit, ('audit_inventory.json',)))
            script = Path(__file__).resolve().parents[1] / 'deploy/archive_pi_deployment.ps1'
            if resume:
                (root / 'output/test-archive-resume/_audit').mkdir(parents=True)
            if verify_only:
                shutil.copytree(audit, root / 'output/test-archive-resume/_audit', dirs_exist_ok=True)
                shutil.copytree(raw, root / 'output/test-archive-resume/bird_images')
            harness = root / 'run.ps1'
            harness.write_text(r'''
$ErrorActionPreference = 'Stop'
function global:ssh {
    if ($env:USB_ARCHIVE_VERIFY_ONLY -eq '1') { throw 'VerifyOnly must not connect to Pi' }
    # Deliberately no stdout, reproducing the observed null-output condition.
    $global:LASTEXITCODE = 0
    if (($args -join ' ') -like '* inventory --source *') {
        Set-Content -LiteralPath (Join-Path $env:USB_ARCHIVE_FIXTURE 'inventoried.txt') -Value ($args -join ' ')
    }
}
function global:scp {
    $remote = [string]$args[-2]
    $destination = [string]$args[-1]
    $fixture = $null
    if ($remote -like '*birdserver/deployment_exports/*') {
        $fixture = Join-Path $env:USB_ARCHIVE_FIXTURE 'audit'
    } elseif ($remote -like '*:/*') {
        $fixture = Join-Path $env:USB_ARCHIVE_FIXTURE 'raw'
        Add-Content -LiteralPath (Join-Path $env:USB_ARCHIVE_FIXTURE 'raw-copy.txt') -Value $remote
    }
    if ($fixture) {
        if ($env:USB_ARCHIVE_RETRY -eq '1' -and !($args -ccontains '-O')) {
            $global:LASTEXITCODE = 1
            return
        }
        if ($args -ccontains '-O') {
            if ($remote.EndsWith('/.')) { throw 'error: unexpected filename: .' }
            $leaf = ($remote -split '/')[-1]
            $destination = Join-Path $destination $leaf
            New-Item -ItemType Directory -Path $destination | Out-Null
        }
        Get-ChildItem -LiteralPath $fixture -Force | Copy-Item -Destination $destination -Recurse -Force
    }
    $global:LASTEXITCODE = 0
}
$options = @{}
if ($env:USB_ARCHIVE_RESUME -eq '1') { $options.ResumeArchiveId = 'test-archive-resume' }
if ($env:USB_ARCHIVE_LEGACY -eq '1') { $options.LegacyScp = $true }
if ($env:USB_ARCHIVE_VERIFY_ONLY -eq '1') { $options.VerifyOnly = $true }
if ($env:USB_ARCHIVE_HISTORY -eq '1') { $options.PreviousVersions = $true }
& $env:USB_ARCHIVE_SCRIPT -DestinationRoot (Join-Path $env:USB_ARCHIVE_FIXTURE 'output') -DeploymentId test-archive @options
''', encoding='utf-8')
            env = dict(os.environ, USB_ARCHIVE_FIXTURE=str(root), USB_ARCHIVE_SCRIPT=str(script),
                       USB_ARCHIVE_RETRY=str(int(retry)), USB_ARCHIVE_RESUME=str(int(resume)),
                       USB_ARCHIVE_LEGACY=str(int(legacy)), USB_ARCHIVE_VERIFY_ONLY=str(int(verify_only)),
                       USB_ARCHIVE_HISTORY=str(int(history)))
            result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-File', str(harness)],
                                    env=env, text=True, capture_output=True, timeout=45)
            snapshots = list((root / 'output').glob('*'))
            if source_root:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('VERIFIED archive:', result.stdout)
                if retry:
                    self.assertIn('Retrying with legacy SCP', result.stdout)
                self.assertEqual((root / 'inventoried.txt').exists(), not resume)
                if history:
                    self.assertIn('--source /mnt/bird-backup/birdhub-backup/previous_versions',
                                  (root / 'inventoried.txt').read_text())
                expected_remote = source_root if legacy else source_root + '/.'
                if not verify_only:
                    self.assertIn(expected_remote, (root / 'raw-copy.txt').read_text())
                self.assertTrue(json.loads((snapshots[0] / '_raw_verification.json').read_text())['verified'])
                self.assertEqual((snapshots[0] / 'bird_images/capture_a.jpg').read_bytes(), b'original image')
                self.assertEqual((snapshots[0] / 'bird_images/crops/a.png').read_bytes(), b'crop')
                self.assertEqual((snapshots[0] / 'bird_images/.hidden').read_bytes(), b'hidden')
                self.assertFalse(list(snapshots[0].glob('.scp-stage-*')))
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('safe absolute capture directory', result.stderr)
                self.assertFalse((root / 'raw-copy.txt').exists())
                self.assertFalse((snapshots[0] / 'SNAPSHOT.json').exists())

    def test_usb_archive_does_not_require_ssh_stdout(self):
        self.run_workflow('/mnt/bird-backup/birdhub-backup/bird_images')

    def test_sd_source_resolved_from_inventory(self):
        self.run_workflow('/home/jrussell/bird_images.before-usb-20261116T183228Z')

    def test_missing_source_root_stops_before_raw_copy(self):
        self.run_workflow(None)

    def test_sftp_failure_retries_legacy_scp(self):
        self.run_workflow('/mnt/bird-backup/birdhub-backup/bird_images', retry=True)

    def test_resume_uses_existing_inventory_without_nesting_directories(self):
        self.run_workflow('/mnt/bird-backup/birdhub-backup/bird_images', resume=True)

    def test_legacy_resume_uses_named_folders_including_nested_and_hidden_files(self):
        self.run_workflow('/mnt/bird-backup/birdhub-backup/bird_images', resume=True, legacy=True)

    def test_verify_only_seals_local_files_without_ssh(self):
        self.run_workflow('/mnt/bird-backup/birdhub-backup/bird_images', resume=True, verify_only=True)

    def test_previous_versions_archive(self):
        self.run_workflow('/mnt/bird-backup/birdhub-backup/previous_versions', history=True, legacy=True)

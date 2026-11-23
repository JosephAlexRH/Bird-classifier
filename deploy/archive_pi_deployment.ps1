param(
    [ValidatePattern('^[a-zA-Z0-9.-]+$')][string]$PiAddress = '192.168.7.128',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$PiUser = 'jrussell',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$DeploymentId = 'garden-feeder-20261116',
    [string]$DestinationRoot = 'E:\Datasets\ESP32Cam_deployments',
    [string]$Note = '',
    [ValidatePattern('^bird_images(?:\.[a-zA-Z0-9_-]+)?$')][string]$SourceFolder = 'bird_images',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$ResumeArchiveId = '',
    [switch]$LegacyScp,
    [switch]$VerifyOnly,
    [switch]$PreviousVersions,
    [switch]$ResumeCapture
)
$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSScriptRoot
$Helper = Join-Path $ProjectDir 'scripts/archive_pi_deployment.py'
$Target = "${PiUser}@${PiAddress}"
$ArchiveId = if ($ResumeArchiveId) { $ResumeArchiveId } else { $DeploymentId + '-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') }
$Snapshot = Join-Path $DestinationRoot $ArchiveId
if ($VerifyOnly -and !$ResumeArchiveId) { throw 'VerifyOnly requires ResumeArchiveId for the downloaded archive.' }
if ($ResumeArchiveId) {
    if (!(Test-Path -LiteralPath $Snapshot -PathType Container)) { throw "Archive to resume does not exist: $Snapshot" }
    if (Test-Path -LiteralPath (Join-Path $Snapshot 'SNAPSHOT.json')) { throw 'Refusing to modify a completed archive.' }
} else {
    if (Test-Path -LiteralPath $Snapshot) { throw "Archive already exists: $Snapshot" }
    New-Item -ItemType Directory -Path $Snapshot | Out-Null
}
function Receive-PiDirectory([string]$RemoteDirectory, [string]$LocalDirectory) {
    New-Item -ItemType Directory -Force -Path $LocalDirectory | Out-Null
    # Retry the same contents path without nesting partial directories. A second
    # attempt uses legacy SCP when the SFTP transfer closes unexpectedly.
    $UseLegacy = $LegacyScp.IsPresent
    if (!$UseLegacy) {
        & scp -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -pr "${Target}:${RemoteDirectory}/." $LocalDirectory
        $UseLegacy = $LASTEXITCODE -ne 0
        if ($UseLegacy) { Write-Warning 'Transfer failed. Retrying with legacy SCP; enter the SSH password again.' }
    }
    if ($UseLegacy) {
        # Legacy SCP rejects a remote basename of '.'. Copy the named folder
        # into fresh staging, then merge its contents into the intended folder.
        $Stage = Join-Path $Snapshot ('.scp-stage-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $Stage | Out-Null
        & scp -O -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -pr "${Target}:${RemoteDirectory}" $Stage
        if ($LASTEXITCODE -ne 0) { throw "Directory transfer failed: $RemoteDirectory. Keep capture stopped and resume this archive." }
        $Leaf = ($RemoteDirectory.TrimEnd('/') -split '/')[-1]
        $Transferred = Join-Path $Stage $Leaf
        if (!(Test-Path -LiteralPath $Transferred -PathType Container)) { throw 'SCP did not produce the expected directory.' }
        Get-ChildItem -LiteralPath $Transferred -Force | Copy-Item -Destination $LocalDirectory -Recurse -Force
        # Delete only this newly created local staging directory, never a Pi
        # path or the archive directory. Check the absolute boundary first.
        $StageAbsolute = [System.IO.Path]::GetFullPath($Stage)
        $SnapshotAbsolute = [System.IO.Path]::GetFullPath($Snapshot).TrimEnd('\') + '\'
        if (!$StageAbsolute.StartsWith($SnapshotAbsolute, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw 'Refusing staging cleanup outside the archive.'
        }
        Remove-Item -LiteralPath $StageAbsolute -Recurse -Force
    }
}
$Stopped = $false
Write-Host "Archive destination: $Snapshot"
if ($VerifyOnly) {
    Write-Host 'Verifying downloaded files locally. No Pi connection or service changes.'
} else {
    Write-Host 'This ends the run by stopping birdhub. Nothing on the Pi will be deleted.'
}
try {
    if (!$VerifyOnly) {
    & ssh $Target 'mkdir -p ~/birdserver/deploy ~/birdserver/deployment_exports'
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the Pi inventory tool.' }
    & scp $Helper "${Target}:birdserver/deploy/archive_pi_deployment.py"
    if ($LASTEXITCODE -ne 0) { throw 'Inventory tool transfer failed.' }
    & ssh -t $Target 'sudo systemctl stop birdhub'
    if ($LASTEXITCODE -ne 0) { throw 'Capture did not stop; no stable archive has been claimed.' }
    $Stopped = $true
    $RequestedSource = if ($PreviousVersions) { '/mnt/bird-backup/birdhub-backup/previous_versions' } else { "/home/$PiUser/$SourceFolder" }
    if (!$ResumeArchiveId) {
        & ssh $Target "if [ -x ~/.venv/bin/python ]; then ~/.venv/bin/python ~/birdserver/deploy/archive_pi_deployment.py inventory --source $RequestedSource --project ~/birdserver --output ~/birdserver/deployment_exports/$ArchiveId; else python3 ~/birdserver/deploy/archive_pi_deployment.py inventory --source $RequestedSource --project ~/birdserver --output ~/birdserver/deployment_exports/$ArchiveId; fi"
        if ($LASTEXITCODE -ne 0) { throw 'Source inventory failed; raw files remain on the Pi.' }
    }
    # Get the canonical source path from the saved inventory, rather than
    # capturing a separate SSH command's stdout (which can be empty).
    Receive-PiDirectory "/home/$PiUser/birdserver/deployment_exports/$ArchiveId" (Join-Path $Snapshot '_audit')
    }
    $SourceInventory = Get-Content -LiteralPath (Join-Path $Snapshot '_audit/source_inventory.json') -Raw | ConvertFrom-Json
    $ResolvedSource = [string]$SourceInventory.source_root
    if ([string]::IsNullOrWhiteSpace($ResolvedSource) -or $ResolvedSource -notmatch '^/[a-zA-Z0-9_./-]+$' -or $ResolvedSource.Contains('/../')) {
        throw 'Source inventory did not contain a safe absolute capture directory. No raw copy attempted.'
    }
    $RawDestination = Join-Path $Snapshot 'bird_images'
    if (!$VerifyOnly) { Receive-PiDirectory $ResolvedSource $RawDestination }
    & python $Helper verify --root (Join-Path $Snapshot 'bird_images') --inventory (Join-Path $Snapshot '_audit/source_inventory.json') --report (Join-Path $Snapshot '_raw_verification.json')
    if ($LASTEXITCODE -ne 0) { throw 'Raw verification FAILED. All source files are still on the Pi.' }
    & python $Helper verify --root (Join-Path $Snapshot '_audit') --inventory (Join-Path $Snapshot '_audit/audit_inventory.json') --report (Join-Path $Snapshot '_audit_verification.json')
    if ($LASTEXITCODE -ne 0) { throw 'Audit/model/code verification FAILED. Keep the Pi unchanged.' }
    $Record = @{
        schema_version = 1
        archive_id = $ArchiveId
        deployment_id = $DeploymentId
        created_at_utc = [DateTime]::UtcNow.ToString('o')
        pi_address = $PiAddress
        source = $ResolvedSource
        source_folder = $SourceFolder
        source_kind = if ($PreviousVersions) { 'usb_previous_versions' } else { 'captures' }
        raw_verified_against_source = $true
        source_files_deleted = $false
        note = $Note
        status = 'verified_raw_archive'
        annotation_policy = 'Raw acquisition is preserved. Model boxes and species are candidates; human reviews and training exports belong in a separate working copy.'
    }
    $Record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Snapshot 'SNAPSHOT.json') -Encoding utf8
    & python $Helper seal --root $Snapshot
    if ($LASTEXITCODE -ne 0) { throw 'Archive sealing failed.' }
    Write-Host "VERIFIED archive: $Snapshot"
    Write-Host 'Keep this raw archive unchanged; make a separate working copy for labelling.'
    Write-Host 'Keep the Pi copy until a second independent archive copy has also passed verification.'
} finally {
    if ($Stopped -and $ResumeCapture) {
        & ssh -t $Target 'sudo systemctl start birdhub'
        if ($LASTEXITCODE -ne 0) { Write-Warning 'Capture did not resume; check birdhub on the Pi.' }
    } elseif ($Stopped) {
        Write-Host 'birdhub remains stopped. Start the next deployment or run sudo systemctl start birdhub when ready.'
    }
}

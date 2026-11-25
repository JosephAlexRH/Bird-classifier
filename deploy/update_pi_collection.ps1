param(
    [ValidatePattern('^[a-zA-Z0-9.-]+$')][string]$PiAddress = '192.168.7.128',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$PiUser = 'jrussell',
    [ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$DeploymentId = 'feeder-test'
)
$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSScriptRoot
$Target = "${PiUser}@${PiAddress}"
$Stamp = Get-Date -Format 'yyyyMMddTHHmmss'
Write-Host 'Backing up Pi server code, then copying the tested collection update. Models and images are kept.'
Write-Host "Pi code backup: ~/birdserver/server-backup-$Stamp"
& ssh $Target "mkdir -p ~/birdserver/deploy && cp -a ~/birdserver/server ~/birdserver/server-backup-$Stamp"
if ($LASTEXITCODE -ne 0) { throw 'Pi backup failed; update stopped.' }
$Files = @('bird_inference.py', 'pi_motion.py', 'raspberry_pi_server.py', 'camera_registry.py',
           'collection.py', 'collection_dashboard.py', 'console_dashboard.py', 'camera_tools.py',
           'hub_dashboard.py', 'hub_settings.py', 'fleet_monitor.py', 'motion_crop.py', 'scene_background.py',
           'visits.py', 'static_boxes.py', 'fleet_stats.py', 'hub_specs.py', 'hub_clock.py')
$Sources = $Files | ForEach-Object { Join-Path $ProjectDir "server/$_" }
& scp @Sources "${Target}:birdserver/server/"
if ($LASTEXITCODE -ne 0) { throw 'Server transfer failed; collection service has not been restarted.' }
& scp (Join-Path $PSScriptRoot 'enable_pi_collection.sh') (Join-Path $PSScriptRoot 'pi_hotspot.sh') (Join-Path $PSScriptRoot 'birdhub_mode.sh') "${Target}:birdserver/deploy/"
if ($LASTEXITCODE -ne 0) { throw 'Installer transfer failed; service has not been restarted.' }
Write-Host 'Enabling collection through SSH. Enter your Pi sudo password if requested.'
& ssh -t $Target "cd ~/birdserver && sudo bash deploy/enable_pi_collection.sh $DeploymentId"
if ($LASTEXITCODE -ne 0) { throw 'Pi service update failed. Check journalctl -u birdhub -n 40 --no-pager on the Pi.' }
Write-Host "Pi dashboard: http://${PiAddress}:5000"
Write-Host "Laptop: python server/laptop_test_server.py --pi-url http://${PiAddress}:5000 --no-advertise"

# Bakes binary terrain (.sbt) for sceneries that have none, several at a time.
#
# Each scenery is baked by its own eu07 process running with -bake, which presents no
# window. Sceneries whose name starts with '$' are skipped: those are Rainsted-created
# overrides and the simulator refuses to compile them.
#
#   .\tools\bake-scenery.ps1 -GameDir "C:\...\MaSzyna"
#   .\tools\bake-scenery.ps1 -GameDir "..." -Parallel 8
#   .\tools\bake-scenery.ps1 -GameDir "..." -Scenery pila.scn,td.scn
#   .\tools\bake-scenery.ps1 -GameDir "..." -WhatIf

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $GameDir,

    # executable to drive, relative to GameDir unless an absolute path is given
    [string] $Exe = "eu07.exe",

    # how many sceneries to bake at once. 0 works it out from free memory, which is what
    # actually limits this: a bake holds its whole region in RAM, anywhere from a few
    # hundred MB to several GB, while the cpu sits mostly idle waiting on disk
    [int] $Parallel = 0,

    # memory to assume a single bake will want
    [double] $PerProcessGB = 2.0,

    # memory to leave to the rest of the machine
    [double] $ReserveGB = 4.0,

    # never run more than this many at once, whatever the memory says
    [int] $MaxParallel = 8,

    # bake only these, instead of everything that lacks a twin
    [string[]] $Scenery,

    # stop starting new bakes once the disk has less than this much room left. a single
    # twin has been seen to pass a gigabyte, so leave headroom
    [int] $MinFreeGB = 10,

    # list what would be baked and stop
    [switch] $WhatIf
)

$ErrorActionPreference = "Stop"

$exePath = if ([System.IO.Path]::IsPathRooted($Exe)) { $Exe } else { Join-Path $GameDir $Exe }
if (-not (Test-Path $exePath)) { throw "no executable at $exePath" }

$sceneryDir = Join-Path $GameDir "scenery"
if (-not (Test-Path $sceneryDir)) { throw "no scenery directory at $sceneryDir" }

if ($Scenery) {
    $pending = $Scenery
}
else {
    $pending = Get-ChildItem -Path $sceneryDir -Filter *.scn -File |
        Where-Object { -not $_.Name.StartsWith('$') } |
        Where-Object { -not (Test-Path (Join-Path $sceneryDir ($_.BaseName + ".sbt"))) } |
        ForEach-Object { $_.Name }
}

if (-not $pending) { Write-Host "nothing to bake"; return }

function Get-FreeRamGB {
    (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
}

if ($Parallel -le 0) {
    $freeRam = Get-FreeRamGB
    $Parallel = [Math]::Floor(($freeRam - $ReserveGB) / $PerProcessGB)
    $Parallel = [Math]::Max(1, [Math]::Min($MaxParallel, $Parallel))
    Write-Host ("wolny RAM {0:N1} GB -> {1} procesow rownolegle" -f $freeRam, $Parallel)
}

Write-Host ("{0} scenery/scenerie do upieczenia, po {1} naraz" -f $pending.Count, $Parallel)

# a twin runs to several hundred MB and occasionally past a gigabyte, so say plainly
# how much room is left before filling somebody's disk
$free = (Get-PSDrive -Name (Split-Path -Qualifier $sceneryDir).TrimEnd(':')).Free
Write-Host ("wolne miejsce na dysku: {0:N1} GB" -f ($free / 1GB))

if ($WhatIf) { $pending | ForEach-Object { "  $_" }; return }

$queue = [System.Collections.Queue]::new(@($pending))
$running = @{}
$failed = @()
$skipped = @()
$done = 0
$started = Get-Date
$driveName = (Split-Path -Qualifier $sceneryDir).TrimEnd(':')

while ($queue.Count -gt 0 -or $running.Count -gt 0) {

    while ($running.Count -lt $Parallel -and $queue.Count -gt 0) {

        if ((Get-PSDrive -Name $driveName).Free / 1GB -lt $MinFreeGB) {
            # better to leave sceneries unbaked than to fill somebody's disk
            Write-Host ("`nprzerwano: zostalo mniej niz {0} GB wolnego miejsca" -f $MinFreeGB) -ForegroundColor Yellow
            $skipped = @($queue.ToArray())
            $queue.Clear()
            break
        }

        # how much a bake wants varies by an order of magnitude between sceneries, so
        # the count worked out up front is only a starting point: hold off on starting
        # another one whenever memory is currently short, rather than push into swap
        if ($running.Count -ge 1 -and (Get-FreeRamGB) -lt ($ReserveGB + $PerProcessGB)) {
            break
        }

        $name = $queue.Dequeue()
        $proc = Start-Process -FilePath $exePath -ArgumentList '-bake', $name `
                              -WorkingDirectory $GameDir -PassThru
        $running[$proc.Id] = @{ Name = $name; Proc = $proc; Start = Get-Date }
        Write-Host ("  start  {0}" -f $name)
    }

    Start-Sleep -Seconds 2

    foreach ($id in @($running.Keys)) {
        $entry = $running[$id]
        if (-not $entry.Proc.HasExited) { continue }

        $running.Remove($id)
        $done++
        $seconds = ((Get-Date) - $entry.Start).TotalSeconds
        $twin = Join-Path $sceneryDir ([System.IO.Path]::GetFileNameWithoutExtension($entry.Name) + ".sbt")

        if ($entry.Proc.ExitCode -eq 0 -and (Test-Path $twin)) {
            $size = (Get-Item $twin).Length / 1MB
            Write-Host ("  ok     {0}  {1:N0}s  {2:N1} MB  [{3}/{4}]" -f $entry.Name, $seconds, $size, $done, $pending.Count)
        }
        else {
            $failed += $entry.Name
            Write-Host ("  FAILED {0}  exit={1}  [{2}/{3}]" -f $entry.Name, $entry.Proc.ExitCode, $done, $pending.Count) -ForegroundColor Red
        }
    }
}

Write-Host ("`nupieczono {0} z {1} w {2:N0}s" -f ($done - $failed.Count), $pending.Count, ((Get-Date) - $started).TotalSeconds)
if ($skipped) {
    Write-Host ("pominieto {0} z braku miejsca; uruchom ponownie po zwolnieniu dysku" -f $skipped.Count) -ForegroundColor Yellow
}
if ($failed) {
    Write-Host "nie powiodly sie:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 1
}

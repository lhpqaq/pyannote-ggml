$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $RootDir "diarization-ggml/build-windows-cpu"
$Jobs = if ($env:JOBS) { $env:JOBS } else { "8" }

Write-Host "[build-windows-cpu] configure: $BuildDir"
cmake -S "$RootDir/diarization-ggml" -B "$BuildDir"

Write-Host "[build-windows-cpu] build -j$Jobs"
cmake --build "$BuildDir" -j $Jobs

Write-Host "[build-windows-cpu] done"

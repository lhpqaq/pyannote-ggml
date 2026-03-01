$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $RootDir "diarization-ggml/build-windows-cuda"
$Jobs = if ($env:JOBS) { $env:JOBS } else { "8" }

Write-Host "[build-windows-cuda] configure: $BuildDir"
cmake -S "$RootDir/diarization-ggml" -B "$BuildDir"

Write-Host "[build-windows-cuda] build -j$Jobs"
cmake --build "$BuildDir" -j $Jobs

Write-Host "[build-windows-cuda] done"

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LocalCacheRoot = Join-Path $ProjectRoot ".cache"
$Targets = @(
    @{
        Source = Join-Path $HOME ".cache\huggingface"
        Destination = Join-Path $LocalCacheRoot "huggingface"
    },
    @{
        Source = Join-Path $HOME ".cache\modelscope"
        Destination = Join-Path $LocalCacheRoot "modelscope"
    }
)

foreach ($pair in $Targets) {
    if (-not (Test-Path $pair.Source)) {
        Write-Host "Skip: $($pair.Source)"
        continue
    }
    New-Item -ItemType Directory -Force -Path $pair.Destination | Out-Null
    robocopy $pair.Source $pair.Destination /E /XO /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    Write-Host "Synced: $($pair.Source) -> $($pair.Destination)"
}

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CacheRoot = Join-Path $ProjectRoot ".cache"
$HFRoot = Join-Path $CacheRoot "huggingface"
$STRoot = Join-Path $CacheRoot "sentence_transformers"
$MSRoot = Join-Path $CacheRoot "modelscope"
$TorchRoot = Join-Path $CacheRoot "torch"

foreach ($dir in @($CacheRoot, $HFRoot, $STRoot, $MSRoot, $TorchRoot)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$env:ESGREVEAL_CACHE_ROOT = $CacheRoot
$env:HF_HOME = $HFRoot
$env:HUGGINGFACE_HUB_CACHE = Join-Path $HFRoot "hub"
$env:TRANSFORMERS_CACHE = Join-Path $HFRoot "transformers"
$env:SENTENCE_TRANSFORMERS_HOME = $STRoot
$env:MODELSCOPE_CACHE = $MSRoot
$env:TORCH_HOME = $TorchRoot

$env:ESGREVEAL_SUMMARY_MODEL = "google/mt5-small"
$env:ESGREVEAL_EMBEDDING_MODEL = "moka-ai/m3e-base"
$env:ESGREVEAL_TABLE_TRANSFORMER_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-all"
$env:ESGREVEAL_LORE_MODEL = "damo/cv_resnet-transformer_table-structure-recognition_lore"
$env:ESGREVEAL_LAYOUT_BACKEND = "geolayoutlm,layoutlmv3,layoutparser"
$env:ESGREVEAL_TABLE_BACKEND = "table_transformer,lore_tsr,fitz"

$GeoLayoutRoot = Join-Path $ProjectRoot "external\AdvancedLiterateMachinery\DocumentUnderstanding\GeoLayoutLM"
if (Test-Path $GeoLayoutRoot) {
    $env:ESGREVEAL_GEOLAYOUTLM_PATH = $GeoLayoutRoot
}

Write-Host "ESGReveal runtime environment configured."
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "CacheRoot   = $CacheRoot"

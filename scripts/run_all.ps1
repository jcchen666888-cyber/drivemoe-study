param(
    [string]$CondaEnv = 'drivemoe-study',
    [string]$ProjectRoot = ''
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Python = (conda run -n $CondaEnv where.exe python | Select-Object -First 1).Trim()
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found in Conda env $CondaEnv" }

& $Python (Join-Path $PSScriptRoot 'verify_assets.py') --root $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw 'Asset verification failed' }
& $Python (Join-Path $PSScriptRoot 'prepare_mini_data.py') `
    --dataset (Join-Path $ProjectRoot 'data\Bench2Drive-Base') `
    --camera-labels (Join-Path $ProjectRoot 'data\labels\camera_labels') `
    --scenario-labels (Join-Path $ProjectRoot 'data\labels\scenario_labels') `
    --output (Join-Path $ProjectRoot 'exp\b2d_action_mini_direct\val')
if ($LASTEXITCODE -ne 0) { throw 'Mini data generation failed' }
$env:CUDA_VISIBLE_DEVICES = '0'
& $Python (Join-Path $PSScriptRoot 'run_minimal_inference.py') --root $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw 'DriveMoE inference failed' }

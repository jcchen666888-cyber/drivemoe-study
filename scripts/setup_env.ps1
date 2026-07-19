param(
    [string]$CondaEnv = 'drivemoe-study',
    [string]$ProjectRoot = ''
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$Upstream = Join-Path $ProjectRoot '_deps\DriveMoE'
if (-not (Test-Path -LiteralPath (Join-Path $Upstream 'pyproject.toml'))) {
    throw "Pinned upstream source is missing: $Upstream"
}

conda create -n $CondaEnv python=3.10 -y
if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed' }
conda run -n $CondaEnv python -m pip install `
    torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
conda run -n $CondaEnv python -m pip install -e $Upstream --no-deps
if ($LASTEXITCODE -ne 0) { throw 'DriveMoE editable installation failed' }
conda run -n $CondaEnv python -m pip install `
    hydra-core==1.3.4 omegaconf==2.3.1 einops==0.8.2 joblib==1.5.3 `
    opencv-python==4.10.0.84 imageio==2.37.3 matplotlib pillow protobuf==3.20.3 `
    tensorflow==2.15.0 tensorflow-datasets==4.9.2 ray==2.56.1 tqdm==4.67.1 `
    transformers==4.49.0 huggingface-hub==0.36.2 scikit-learn==1.7.2
if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed' }
conda run -n $CondaEnv python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke test failed' }

param([string]$ProjectRoot = '')

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$ArchiveRoot = Join-Path $ProjectRoot 'artifacts\bench2drive_archives'
$DataRoot = Join-Path $ProjectRoot 'data\Bench2Drive-Base'
New-Item -ItemType Directory -Force -Path $DataRoot | Out-Null
$Archives = Get-ChildItem -LiteralPath $ArchiveRoot -Filter '*.tar.gz'
if ($Archives.Count -ne 3) { throw "Expected exactly 3 route archives, got $($Archives.Count)" }
foreach ($Archive in $Archives) {
    tar -xzf $Archive.FullName -C $DataRoot
    if ($LASTEXITCODE -ne 0) { throw "Extraction failed: $($Archive.Name)" }
}
$Routes = Get-ChildItem -LiteralPath $DataRoot -Directory
if ($Routes.Count -ne 3) { throw "Expected exactly 3 extracted routes, got $($Routes.Count)" }
Write-Output "PASS: extracted $($Routes.Count) routes"

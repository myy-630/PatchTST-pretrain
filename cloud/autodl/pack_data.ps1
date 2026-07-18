$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\.."
$DataRoot = Join-Path $ProjectRoot "kaggle_upload\shd-af-clean-data"
$OutDir = Join-Path $ProjectRoot "dist"
$Archive = Join-Path $OutDir "shd-af-clean-data.tar.gz"

if (!(Test-Path -LiteralPath $DataRoot)) {
    throw "Data root not found: $DataRoot"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Creating archive:"
Write-Host "  source: $DataRoot"
Write-Host "  output: $Archive"
Write-Host "This can take a long time for ~25 GiB of ECG data."

Push-Location (Split-Path $DataRoot -Parent)
try {
    tar -czf $Archive "shd-af-clean-data"
}
finally {
    Pop-Location
}

Get-Item -LiteralPath $Archive | Select-Object FullName, Length


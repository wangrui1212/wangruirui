$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv-build"
$DistDir = Join-Path $ProjectRoot "dist"
$BuildDir = Join-Path $ProjectRoot "build"
$AppName = "TDAS_DAT_Viewer"
$AppDistDir = Join-Path $DistDir $AppName
$ZipPath = Join-Path $DistDir "$AppName`_portable.zip"
$SpecPath = Join-Path $ProjectRoot "$AppName.spec"
$SourcePath = Join-Path $ProjectRoot "src\tdas_dat_viewer.py"
$RequirementsPath = Join-Path $ProjectRoot "requirements-build.txt"

if (-not (Test-Path $SourcePath)) {
    throw "Source file not found: $SourcePath"
}

if (-not (Test-Path $RequirementsPath)) {
    throw "Requirements file not found: $RequirementsPath"
}

Write-Host "Project root: $ProjectRoot"
Write-Host "Preparing build virtual environment..."

if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
}

$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Build Python not found: $PythonExe"
}

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r $RequirementsPath

Write-Host "Cleaning previous build outputs..."
if (Test-Path $BuildDir) {
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}
if (Test-Path $AppDistDir) {
    Remove-Item -LiteralPath $AppDistDir -Recurse -Force
}
if (Test-Path $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
if (Test-Path $SpecPath) {
    Remove-Item -LiteralPath $SpecPath -Force
}

Write-Host "Building portable application..."
& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name $AppName `
    --hidden-import pyqtgraph.opengl `
    --hidden-import OpenGL `
    $SourcePath

$ExePath = Join-Path $AppDistDir "$AppName.exe"
if (-not (Test-Path $ExePath)) {
    throw "Build finished but executable was not found: $ExePath"
}

Write-Host "Creating portable zip..."
Compress-Archive -Path $AppDistDir -DestinationPath $ZipPath -Force

Write-Host ""
Write-Host "Portable build complete."
Write-Host "Folder: $AppDistDir"
Write-Host "Zip:    $ZipPath"
Write-Host "Run:    $ExePath"

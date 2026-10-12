param([switch]$SkipDependencyInstall, [string]$ReleaseFolder = "")

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv-build"
$DistDir = Join-Path $ProjectRoot "dist"
if ($ReleaseFolder) {
    if ($ReleaseFolder -notmatch '^[a-zA-Z0-9_-]+$') { throw "Invalid release folder name" }
    $DistDir = Join-Path $DistDir $ReleaseFolder
}
$BuildDir = Join-Path $ProjectRoot "build\viewer-portable"
$AppName = "TDAS_DAT_Viewer"
$AppDistDir = Join-Path $DistDir $AppName
$ZipPath = Join-Path $DistDir "$AppName`_portable.zip"
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
    if ($LASTEXITCODE -ne 0) { throw "Failed to create build environment" }
}

$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Build Python not found: $PythonExe"
}

if (-not $SkipDependencyInstall) {
    & $PythonExe -m pip install -r $RequirementsPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to install build dependencies" }
}

# Keep other applications' build output and the checked-in spec untouched.
# PyInstaller replaces only this validated application output directory.
if (-not [IO.Path]::GetFullPath($AppDistDir).StartsWith(([IO.Path]::GetFullPath((Join-Path $ProjectRoot "dist")) + "\"), [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unexpected application output directory"
}
New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null

$OriginalBuildPath = $env:PATH
# External tools can ship incompatible DLLs with Windows system DLL names
# (notably Poppler's ICU). Never let those directories affect collection.
$env:PATH = "$VenvDir\Scripts;$env:SystemRoot\System32;$env:SystemRoot"
try {

Write-Host "Building portable application..."
& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name $AppName `
    --distpath $DistDir `
    --workpath $BuildDir `
    --specpath $BuildDir `
    --hidden-import das_report `
    --collect-all scipy `
    --collect-data pyqtgraph `
    --hidden-import pyqtgraph.opengl `
    --hidden-import OpenGL `
    $SourcePath
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$ExePath = Join-Path $AppDistDir "$AppName.exe"
if (-not (Test-Path $ExePath)) {
    throw "Build finished but executable was not found: $ExePath"
}

Write-Host "Checking executable startup (main viewer and report workspace)..."
$SmokeProcess = Start-Process -FilePath $ExePath -ArgumentList "--smoke-test" -WorkingDirectory $AppDistDir -WindowStyle Hidden -PassThru
if (-not $SmokeProcess.WaitForExit(60000)) {
    Stop-Process -Id $SmokeProcess.Id -Force
    throw "Portable startup timed out; ZIP was not published"
}
$SmokeProcess.Refresh()
if ($SmokeProcess.ExitCode -ne 0) { throw "Portable startup failed: $($SmokeProcess.ExitCode)" }

Write-Host "Creating portable zip..."
Compress-Archive -Path $AppDistDir -DestinationPath $ZipPath -Force
} finally {
    $env:PATH = $OriginalBuildPath
}

Write-Host ""
Write-Host "Portable build complete."
Write-Host "Folder: $AppDistDir"
Write-Host "Zip:    $ZipPath"
Write-Host "Run:    $ExePath"

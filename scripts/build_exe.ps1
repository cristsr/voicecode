<#
.SYNOPSIS
    Empaqueta VoiceCode (tray_app.py) como app standalone con PyInstaller.

.DESCRIPTION
    Genera dist/<Name>/<Name>.exe sin ventana de consola (--noconsole), en
    modo --onedir (no --onefile): arranca más rápido porque no tiene que
    descomprimirse en una carpeta temporal en cada ejecución, y además hace
    que las DLLs de cuBLAS/cuDNN queden en una ubicación fija y predecible
    junto al .exe (necesario para que utils.platform.add_nvidia_dll_directories()
    las encuentre cuando el proceso está "frozen").

    Las carpetas nvidia/cublas/bin y nvidia/cudnn/bin normalmente no las
    detecta el análisis automático de PyInstaller (ctranslate2 las carga en
    runtime, no via import de Python) - por eso se agregan explícitamente
    con --collect-binaries.

    config.toml se copia al lado del .exe para que quede editable sin
    reconstruir el bundle (NO se empaqueta adentro a propósito).

    Usa --clean para forzar a PyInstaller a tirar su caché intermedia
    (build/) antes de reconstruir. Sin esto, mover o renombrar la carpeta
    del proyecto entre builds puede dejar esa caché parcialmente
    desincronizada y producir un bundle que carga el modelo bien pero
    falla recién al primer uso real de la GPU (cublas64_12.dll not found),
    algo que no se nota probando solo que el proceso arranca.

.PARAMETER Entry
    Script de entrada a empaquetar. Default: tray_app.py

.PARAMETER Name
    Nombre del ejecutable resultante. Default: VoiceCode

.EXAMPLE
    ./scripts/build_exe.ps1
#>
param(
    [string]$Entry = "tray_app.py",
    [string]$Name = "VoiceCode"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Host "pyinstaller no está instalado. Instalando..."
    pip install pyinstaller
}

pyinstaller --noconsole --noconfirm --clean --name $Name `
    --collect-binaries nvidia.cublas `
    --collect-binaries nvidia.cudnn `
    $Entry

if ($LASTEXITCODE -ne 0) {
    # pyinstaller is a native exe - a non-zero exit code here is NOT a
    # PowerShell terminating error, so $ErrorActionPreference alone would
    # not stop the script and it would go on to report success wrongly.
    throw "pyinstaller failed with exit code $LASTEXITCODE - build did not complete."
}

$AppDir = Join-Path (Join-Path $ProjectRoot "dist") $Name
$ConfigSrc = Join-Path $ProjectRoot "config.toml"
$ConfigDst = Join-Path $AppDir "config.toml"

Copy-Item -Path $ConfigSrc -Destination $ConfigDst -Force

Write-Host ""
Write-Host "Build lista en: $AppDir\$Name.exe"
Write-Host "config.toml copiado junto al ejecutable - editalo ahi, no dentro del bundle."
Write-Host "Los logs (cuando corra --noconsole) van a: $AppDir\voicecode.log"

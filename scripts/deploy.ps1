<#
.SYNOPSIS
    Copia el build de dist/<Name> a su ubicación final en Program Files.

.DESCRIPTION
    Separa "compilar" (build_exe.ps1, sin privilegios especiales, se corre
    seguido mientras se itera) de "desplegar" (este script, requiere
    Administrador porque el destino está protegido igual que el C:\Program
    Files real - solo Administrators/SYSTEM pueden escribir ahí).

    Si el destino ya existe, se reemplaza por completo con el build nuevo.

.PARAMETER Name
    Nombre de la carpeta de la app. Default: VoiceCode

.PARAMETER DeployRoot
    Carpeta raíz donde desplegar. Default: E:\Program Files

.EXAMPLE
    # Desde una PowerShell abierta como Administrador:
    ./scripts/deploy.ps1
#>
param(
    [string]$Name = "VoiceCode",
    [string]$DeployRoot = "E:\Program Files"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceDir = Join-Path (Join-Path $ProjectRoot "dist") $Name
$TargetDir = Join-Path $DeployRoot $Name

if (-not (Test-Path $SourceDir)) {
    throw "No se encontró '$SourceDir'. Corré scripts/build_exe.ps1 primero."
}

$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "'$DeployRoot' tiene permisos restringidos (como el Program Files real) - " +
          "este script necesita correr desde una PowerShell abierta como Administrador."
}

if (Test-Path $TargetDir) {
    Write-Host "Ya existe '$TargetDir' - se reemplaza con el build nuevo."
    Remove-Item -Path $TargetDir -Recurse -Force
}

robocopy $SourceDir $TargetDir /E /MOVE /R:3 /W:2 /NFL /NDL /NP | Out-Null
if ($LASTEXITCODE -ge 8) {
    throw "robocopy falló con código $LASTEXITCODE al desplegar a '$TargetDir'."
}

Write-Host ""
Write-Host "Desplegado en: $TargetDir\$Name.exe"
Write-Host "Para registrar el arranque automático: ./scripts/register_task.ps1"

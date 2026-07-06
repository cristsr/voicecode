<#
.SYNOPSIS
    Registra VoiceCode como tarea de Task Scheduler que arranca al iniciar sesión.

.DESCRIPTION
    Crea una tarea que corre SOLO cuando el usuario actual tiene sesión iniciada
    (LogonType Interactive) - necesario porque VoiceCode usa el teclado global y
    el portapapeles de la sesión interactiva, algo que un servicio de Windows real
    (Sesión 0) no puede hacer.

.PARAMETER ExePath
    Ruta al .exe ya desplegado (ver scripts/deploy.ps1). Default: E:\Program Files\VoiceCode\VoiceCode.exe

.PARAMETER TaskName
    Nombre de la tarea en Task Scheduler. Default: VoiceCode

.EXAMPLE
    ./scripts/register_task.ps1
.EXAMPLE
    ./scripts/register_task.ps1 -ExePath "C:\Tools\VoiceCode\VoiceCode.exe"
#>
param(
    [string]$ExePath = "E:\Program Files\VoiceCode\VoiceCode.exe",
    [string]$TaskName = "VoiceCode"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) {
    throw "No se encontró el ejecutable en '$ExePath'. Corré scripts/build_exe.ps1 y luego scripts/deploy.ps1 primero, o pasá -ExePath explícito."
}

$WorkingDir = Split-Path -Parent $ExePath

$Action = New-ScheduledTaskAction -Execute $ExePath -WorkingDirectory $WorkingDir
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
    -Principal $Principal -Settings $Settings -Force | Out-Null

Write-Host "Tarea '$TaskName' registrada."
Write-Host "Se ejecutará '$ExePath' la próxima vez que $env:USERNAME inicie sesión."
Write-Host ""
Write-Host "Para arrancarla ahora sin reiniciar sesión:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"

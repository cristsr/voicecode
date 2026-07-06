<#
.SYNOPSIS
    Elimina la tarea de Task Scheduler registrada por register_task.ps1.

.PARAMETER TaskName
    Nombre de la tarea a eliminar. Default: VoiceCode

.EXAMPLE
    ./scripts/unregister_task.ps1
#>
param(
    [string]$TaskName = "VoiceCode"
)

$ErrorActionPreference = "Stop"

$Existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $Existing) {
    Write-Host "No existe una tarea llamada '$TaskName'. Nada que hacer."
    return
}

Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false

Write-Host "Tarea '$TaskName' eliminada."
Write-Host "Si VoiceCode sigue corriendo, cerralo desde el ícono de la bandeja (Salir)."

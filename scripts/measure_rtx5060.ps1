# Prophet Studio : mesure du duo reel (S1 classifieur + S2 raisonnement) sur la machine cible, en une commande.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\measure_rtx5060.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\measure_rtx5060.ps1 -Rapide -Label "pilote 581.57"
#
# Trouve le Prophet Studio en marche (core.json du dossier de donnees : -DataDir, sinon PROPHET_HOME, sinon
# %LOCALAPPDATA%\ProphetStudio), demarre les modeles s'ils sont arretes, lance eval\measure_duo.py avec l'environnement
# Python installe (<donnees>\app-venv des scripts d'installation, sinon <donnees>\venv de l'application de bureau, sinon
# `uv run` depuis ce depot), puis affiche le fichier de resultats (eval\results\duo-<date>.json) et le tableau a coller
# dans eval\SCOREBOARD.md (copie aussi dans le presse-papiers). Les autres arguments vont tels quels a measure_duo
# (ex. --n-froid 8). Duree : ~5 minutes (-Rapide : ~2 minutes).
# Fichier volontairement en ASCII : Windows PowerShell 5.1 lit les scripts sans BOM en ANSI.
[CmdletBinding()]
param(
    [string]$DataDir,
    [switch]$Rapide,
    [switch]$SansTours,
    [string]$Label,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot

if (-not $DataDir) {
    if ($env:PROPHET_HOME) { $DataDir = $env:PROPHET_HOME }
    elseif ($env:LOCALAPPDATA) { $DataDir = Join-Path $env:LOCALAPPDATA 'ProphetStudio' }
    else { $DataDir = Join-Path (Join-Path $HOME '.local') 'ProphetStudio' }
}
$core = @((Join-Path $DataDir 'core.json'), (Join-Path (Join-Path $DataDir 'demo') 'core.json')) |
    Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $core) {
    Write-Host "Prophet Studio ne tourne pas : aucun core.json dans $DataDir." -ForegroundColor Yellow
    Write-Host "Lancez Prophet Studio (menu Demarrer ou application de bureau), attendez l'ecran principal, puis relancez."
    exit 2
}
Write-Host "Prophet Studio : $core"

$measureArgs = @('-m', 'eval.measure_duo', '--studio', '--data-dir', $DataDir, '--demarrer')
if ($Rapide) { $measureArgs += '--rapide' }
if ($SansTours) { $measureArgs += '--sans-tours' }
if ($Label) { $measureArgs += @('--label', $Label) }
if ($Rest) { $measureArgs += $Rest }

$python = $null
foreach ($venv in 'app-venv', 'venv') {
    $candidate = Join-Path (Join-Path (Join-Path $DataDir $venv) 'Scripts') 'python.exe'
    if (Test-Path -LiteralPath $candidate) { $python = $candidate; break }
}
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $python -and -not $uv) {
    Write-Host "Aucun Python : ni $DataDir\app-venv, ni $DataDir\venv, ni uv. Installez Prophet Studio (installer\install.ps1)." -ForegroundColor Red
    exit 2
}

$started = Get-Date
$code = 1
Push-Location -LiteralPath $Root
# les programmes externes ecrivent sur stderr sans que ce soit une erreur (Windows PowerShell 5.1)
$ErrorActionPreference = 'Continue'
try {
    if ($python) {
        Write-Host "Python : $python"
        & $python @measureArgs
    } else {
        Write-Host 'Python : uv run (depot)'
        & uv run python @measureArgs
    }
    $code = $LASTEXITCODE
} finally {
    $ErrorActionPreference = 'Stop'
    Pop-Location
}

$results = Join-Path (Join-Path $Root 'eval') 'results'
$latest = $null
if (Test-Path -LiteralPath $results) {
    $latest = Get-ChildItem -LiteralPath $results -Filter 'duo-*.json' | Where-Object { $_.LastWriteTime -ge $started } |
        Sort-Object LastWriteTime | Select-Object -Last 1
}
if ($latest) {
    try {
        $md = (Get-Content -LiteralPath $latest.FullName -Raw -Encoding UTF8 | ConvertFrom-Json).markdown -join "`r`n"
        Set-Clipboard -Value $md
        Write-Host ''
        Write-Host "Tableau copie dans le presse-papiers : collez-le dans eval\SCOREBOARD.md (section 'Duo reel')." -ForegroundColor Green
    } catch {
        Write-Host "Tableau non copie dans le presse-papiers : $($_.Exception.Message)"
    }
    Write-Host "Fichier de resultats : $($latest.FullName)" -ForegroundColor Green
}
exit $code

# Prophet Studio : installation en une commande pour Windows 10/11 x64 (interface dans le navigateur).
#
#   irm https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/main/installer/install.ps1 | iex
#
#   # avec des parametres (version precise, sans raccourcis...) :
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/speed25200-cyber/LLM-and-Classifier/main/installer/install.ps1))) -Ref v0.2.0
#
#   # depuis un depot clone :
#   powershell -ExecutionPolicy Bypass -File .\installer\install.ps1 -Source .
#
# Relancer le script = mettre a jour. Tout va dans %LOCALAPPDATA%\ProphetStudio (code, environnement Python,
# modeles) ; en dehors : uv s'il manque (%USERPROFILE%\.local\bin) et les raccourcis "Prophet Studio"
# du menu Demarrer et du Bureau. Compatible Windows PowerShell 5.1 et PowerShell 7.
# Fichier volontairement en ASCII : Windows PowerShell 5.1 lit les scripts sans BOM en ANSI.
# Sous `irm | iex`, les valeurs par defaut d'un bloc param ne sont pas appliquees : elles sont
# resolues dans Install-ProphetStudio (parametres, sinon PROPHET_REF / PROPHET_SOURCE / PROPHET_REPO /
# PROPHET_HOME / PROPHET_NO_SHORTCUT, sinon valeurs par defaut).
[CmdletBinding()]
param(
    [string]$Ref,
    [string]$Source,
    [string]$Repo,
    [string]$DataDir,
    [switch]$NoShortcut
)

$PythonVersion = '3.11'

# ---- affichage ----------------------------------------------------------------------------------------------------
function Write-Step([string]$Text) { Write-Host ''; Write-Host "==> $Text" -ForegroundColor Cyan }
function Write-Info([string]$Text) { Write-Host "    $Text" }
function Write-Ok([string]$Text) { Write-Host '    OK ' -ForegroundColor Green -NoNewline; Write-Host $Text }
function Write-Attention([string]$Text) { Write-Host '    ATTENTION ' -ForegroundColor Yellow -NoNewline; Write-Host $Text }

# Lance un programme externe et echoue proprement si son code de sortie n'est pas 0.
# * la sortie va a l'ecran (Out-Host) et jamais dans la valeur de retour de la fonction appelante ;
# * Windows PowerShell 5.1 transforme la sortie d'erreur des programmes en erreurs PowerShell : on
#   repasse donc en 'Continue' le temps de l'appel.
function Invoke-Native {
    param([Parameter(Mandatory)][string]$FilePath, [string[]]$Arguments = @(), [string]$What = $FilePath, [switch]$Quiet)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($Quiet) { & $FilePath @Arguments | Out-Null } else { & $FilePath @Arguments | Out-Host }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) { throw "$What a echoue (code $code)" }
}

function Save-Url([string]$Url, [string]$OutFile) {
    Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
}

# ---- systeme ------------------------------------------------------------------------------------------------------
function Get-DefaultDataDir {
    if ($env:LOCALAPPDATA) { return (Join-Path $env:LOCALAPPDATA 'ProphetStudio') }
    return (Join-Path (Join-Path $HOME '.local') 'ProphetStudio')
}

function Test-System {
    Write-Step 'Verification du systeme'
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        throw 'ce script est pour Windows (Linux et macOS : installer/install.sh)'
    }
    $version = [Environment]::OSVersion.Version
    if ($version.Major -lt 10) { throw "Windows 10 ou 11 requis (version detectee : $version)" }
    $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows 64 bits requis' }
    if ($arch -eq 'ARM64') {
        Write-Attention 'Windows ARM64 : pas de CUDA, les modeles tourneront sur le processeur (lent).'
    } elseif ($arch -ne 'AMD64') {
        Write-Attention "architecture $arch non testee"
    }
    Write-Ok "Windows $($version.Major).$($version.Minor) build $($version.Build), $arch, PowerShell $($PSVersionTable.PSVersion)"
}

# Version du pilote NVIDIA a partir de celle de Windows : 32.0.15.7283 -> 572.83 (5 derniers chiffres).
function ConvertTo-NvidiaDriverVersion([string]$WindowsVersion) {
    $digits = ($WindowsVersion -split '\.' | Select-Object -Last 2) -join ''
    if ($digits.Length -lt 5) { return $null }
    $last = $digits.Substring($digits.Length - 5)
    return '{0}.{1}' -f [int]$last.Substring(0, 3), $last.Substring(3)
}

# Une RTX 50xx (Blackwell, sm_120) exige CUDA 12.8, donc un pilote >= 570.
function Test-BlackwellDriver([string]$Name, [string]$Driver) {
    if ($Name -notmatch 'RTX\s*50\d\d') { return $true }
    $major = 0
    if (-not [int]::TryParse(($Driver -split '\.')[0], [ref]$major)) { return $true }
    return ($major -ge 570)
}

function Get-NvidiaGpus {
    $smi = @(
        (Get-Command nvidia-smi -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source),
        (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'),
        (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
    if ($smi) {
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try { $lines = & $smi --query-gpu=name,driver_version --format=csv,noheader 2>$null } finally { $ErrorActionPreference = $previous }
        if ($LASTEXITCODE -eq 0 -and $lines) {
            return @($lines | ForEach-Object {
                    $parts = $_ -split ','
                    [pscustomobject]@{ Name = $parts[0].Trim(); Driver = $parts[1].Trim() }
                })
        }
    }
    # repli : WMI (pilote present mais nvidia-smi absent du PATH)
    return @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like '*NVIDIA*' } |
            ForEach-Object { [pscustomobject]@{ Name = $_.Name; Driver = (ConvertTo-NvidiaDriverVersion $_.DriverVersion) } })
}

function Test-Gpu {
    $gpus = @(Get-NvidiaGpus)
    if ($gpus.Count -eq 0) {
        Write-Attention 'aucun GPU NVIDIA detecte : les modeles tourneront sur le processeur, bien plus lentement.'
        return
    }
    foreach ($gpu in $gpus) {
        Write-Ok "GPU : $($gpu.Name) (pilote $($gpu.Driver))"
        if (-not (Test-BlackwellDriver $gpu.Name $gpu.Driver)) {
            Write-Attention "carte RTX 50xx (Blackwell) avec le pilote $($gpu.Driver) : il faut un pilote >= 570 (CUDA 12.8)."
            Write-Attention 'Mettez le pilote a jour : https://www.nvidia.com/Download/index.aspx'
        }
    }
}

# ---- uv -----------------------------------------------------------------------------------------------------------
function Find-Uv {
    $candidates = @(
        (Get-Command uv.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source),
        (Get-Command uv -ErrorAction SilentlyContinue | Where-Object { $_.CommandType -eq 'Application' } | Select-Object -First 1 -ExpandProperty Source)
    )
    if ($env:USERPROFILE) {
        $candidates += (Join-Path $env:USERPROFILE '.local\bin\uv.exe'), (Join-Path $env:USERPROFILE '.cargo\bin\uv.exe')
    }
    return ($candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1)
}

function Get-Uv {
    Write-Step 'Gestionnaire Python uv'
    $uv = Find-Uv
    if (-not $uv) {
        Write-Info 'uv absent : installation depuis https://astral.sh/uv/install.ps1'
        Invoke-Native -FilePath 'powershell.exe' -What "l'installation de uv" -Arguments @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', 'irm https://astral.sh/uv/install.ps1 | iex')
        $uv = Find-Uv
        if (-not $uv) { throw "uv reste introuvable apres son installation" }
    }
    $binDir = Split-Path -Parent $uv
    if (($env:Path -split ';') -notcontains $binDir) { $env:Path = "$binDir;$env:Path" }
    Write-Ok "uv : $uv ($(& $uv --version))"
    return $uv
}

# ---- sources ------------------------------------------------------------------------------------------------------
function Get-Sources([string]$Tmp, [string]$Source, [string]$Repo, [string]$Ref) {
    if ($Source) {
        Write-Step 'Sources : depot local'
        $stage = (Resolve-Path -LiteralPath $Source).Path
    } else {
        Write-Step "Telechargement de Prophet Studio ($Repo @ $Ref)"
        $zip = Join-Path $Tmp 'src.zip'
        try {
            Save-Url "https://codeload.github.com/$Repo/zip/$Ref" $zip
        } catch {
            throw "telechargement impossible (reference '$Ref' inconnue ou reseau indisponible) : $($_.Exception.Message)"
        }
        $extract = Join-Path $Tmp 'src'
        Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force
        $stage = (Get-ChildItem -LiteralPath $extract -Directory | Select-Object -First 1).FullName
    }
    if (-not ($stage -and (Test-Path -LiteralPath (Join-Path $stage 'pyproject.toml')) -and (Test-Path -LiteralPath (Join-Path $stage 'prophet_studio')))) {
        throw "$stage ne contient pas Prophet Studio (pyproject.toml + prophet_studio)"
    }
    Write-Ok $stage
    return $stage
}

function Test-RunningInstance([string]$Data) {
    $info = Join-Path $Data 'core.json'
    if (-not (Test-Path -LiteralPath $info)) { return }
    try {
        $url = (Get-Content -LiteralPath $info -Raw | ConvertFrom-Json).url
        $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 2
        if ($health.app -eq 'prophet-studio') {
            Write-Attention "Prophet Studio tourne encore ($url) : fermez-le si la mise a jour echoue, puis redemarrez-le ensuite."
        }
    } catch {
        return
    }
}

# Copie le code dans <data>\app (via app.new pour ne jamais laisser une installation a moitie copiee).
# Renvoie le chemin d'une archive de l'interface a extraire apres uv sync, ou $null.
function Copy-App([string]$Stage, [string]$Data, [string]$Tmp, [string]$Repo) {
    Write-Step "Copie de l'application"
    $app = Join-Path $Data 'app'
    $new = Join-Path $Data 'app.new'
    if (Test-Path -LiteralPath $new) { Remove-Item -LiteralPath $new -Recurse -Force }
    New-Item -ItemType Directory -Path $new -Force | Out-Null
    foreach ($item in 'pyproject.toml', 'uv.lock', 'README.md', 'LICENSE', 'jev_clone', 'prophet_studio') {
        $path = Join-Path $Stage $item
        if (Test-Path -LiteralPath $path) { Copy-Item -LiteralPath $path -Destination $new -Recurse -Force }
    }
    Get-ChildItem -LiteralPath $new -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force

    # Interface web : deja construite, sinon ui\dist, sinon construite avec npm, sinon telechargee.
    $web = Join-Path (Join-Path $new 'prophet_studio') 'web'
    $webZip = $null
    $uiDist = Join-Path (Join-Path $Stage 'ui') 'dist'
    if (Test-Path -LiteralPath (Join-Path $web 'index.html')) {
        Write-Ok 'interface deja construite'
    } elseif (Test-Path -LiteralPath (Join-Path $uiDist 'index.html')) {
        Copy-Item -LiteralPath $uiDist -Destination $web -Recurse -Force
        Write-Ok 'interface copiee depuis ui\dist'
    } else {
        $npm = Get-Command npm.cmd, npm -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
        $uiSrc = Join-Path $Stage 'ui'
        if ($npm -and (Test-Path -LiteralPath (Join-Path $uiSrc 'package.json'))) {
            Write-Info "construction de l'interface avec npm (1 a 2 minutes)"
            $uiNew = Join-Path $new 'ui'
            New-Item -ItemType Directory -Path $uiNew -Force | Out-Null
            Get-ChildItem -LiteralPath $uiSrc -Force | Where-Object { $_.Name -notin 'node_modules', 'dist' } |
                Copy-Item -Destination $uiNew -Recurse -Force
            Push-Location $uiNew
            try {
                Invoke-Native -FilePath $npm -Arguments @('ci', '--no-audit', '--no-fund', '--loglevel=error') -What 'npm ci'
                Invoke-Native -FilePath $npm -Arguments @('run', 'build', '--silent') -What 'npm run build'
                Write-Ok 'interface construite'
            } catch {
                Write-Attention "construction de l'interface echouee : $($_.Exception.Message)"
            } finally {
                Pop-Location
            }
            Remove-Item -LiteralPath $uiNew -Recurse -Force -ErrorAction SilentlyContinue
        }
        if (-not (Test-Path -LiteralPath (Join-Path $web 'index.html'))) {
            $zip = Join-Path $Tmp 'web.zip'
            try {
                Save-Url "https://github.com/$Repo/releases/latest/download/prophet-studio-web.zip" $zip
                $webZip = $zip
                Write-Ok 'interface precompilee telechargee'
            } catch {
                Write-Attention "interface web introuvable : installez Node.js 20+ puis relancez, ou utilisez l'application de bureau."
            }
        }
    }

    $icon = Join-Path $Stage 'desktop\src-tauri\icons\icon.ico'
    if (Test-Path -LiteralPath $icon) { Copy-Item -LiteralPath $icon -Destination (Join-Path $Data 'prophet-studio.ico') -Force }

    if (Test-Path -LiteralPath $app) { Remove-Item -LiteralPath $app -Recurse -Force }
    Move-Item -LiteralPath $new -Destination $app
    Write-Ok $app
    return $webZip
}

# ---- environnement Python -----------------------------------------------------------------------------------------
function Sync-Environment([string]$Uv, [string]$Data) {
    Write-Step 'Environnement Python (uv sync, 1 a 3 minutes la premiere fois)'
    $previous = $env:UV_PROJECT_ENVIRONMENT
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $Data 'app-venv'
    try {
        Invoke-Native -FilePath $Uv -What 'uv sync' -Arguments @(
            'sync', '--project', (Join-Path $Data 'app'), '--extra', 'studio', '--python', $PythonVersion)
    } finally {
        $env:UV_PROJECT_ENVIRONMENT = $previous
    }
}

function Get-VenvTool([string]$Data, [string]$Name) {
    $venv = Join-Path $Data 'app-venv'
    foreach ($candidate in (Join-Path (Join-Path $venv 'Scripts') "$Name.exe"), (Join-Path (Join-Path $venv 'bin') $Name)) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

# Sous pythonw.exe, sys.stdout et sys.stderr valent None et uvicorn plante en configurant ses journaux :
# ce lanceur les redirige vers logs\launcher.log avant de demarrer le coeur.
function Write-PywLauncher([string]$Data) {
    $launcher = Join-Path $Data 'prophet-studio.pyw'
    $code = @'
# Lanceur de Prophet Studio sans console (pythonw.exe), cree par installer/install.ps1.
# pythonw laisse sys.stdout / sys.stderr a None : on les redirige vers logs/launcher.log.
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(here, "logs"), exist_ok=True)
log = open(os.path.join(here, "logs", "launcher.log"), "w", encoding="utf-8", buffering=1)
if sys.stdout is None:
    sys.stdout = log
if sys.stderr is None:
    sys.stderr = log
sys.argv = ["prophet-studio", *sys.argv[1:]]
runpy.run_module("prophet_studio", run_name="__main__", alter_sys=True)
'@
    Set-Content -LiteralPath $launcher -Value $code -Encoding Ascii
    return $launcher
}

function New-Shortcuts([string]$Data, [string]$Launcher) {
    Write-Step 'Raccourcis'
    $pythonw = Get-VenvTool $Data 'pythonw'
    $target = $pythonw
    if (-not $target) {
        $target = Get-VenvTool $Data 'python'
        Write-Attention 'pythonw.exe absent : le raccourci ouvrira une console.'
    }
    $icon = Join-Path $Data 'prophet-studio.ico'
    $shell = New-Object -ComObject WScript.Shell
    $folders = @([Environment]::GetFolderPath('Programs'), [Environment]::GetFolderPath('DesktopDirectory'))
    foreach ($folder in $folders) {
        if (-not $folder) { continue }
        $path = Join-Path $folder 'Prophet Studio.lnk'
        $shortcut = $shell.CreateShortcut($path)
        $shortcut.TargetPath = $target
        $shortcut.Arguments = '"' + $Launcher + '"'
        $shortcut.WorkingDirectory = $Data
        $shortcut.Description = 'Prophet Studio : Bonsai 2 27B + classifieur, en local'
        if (Test-Path -LiteralPath $icon) { $shortcut.IconLocation = "$icon,0" }
        $shortcut.Save()
        Write-Ok $path
    }
}

# ---- programme principal ------------------------------------------------------------------------------------------
function Install-ProphetStudio {
    param([string]$Ref, [string]$Source, [string]$Repo, [string]$DataDir, [bool]$NoShortcut)
    if (-not $Ref) { $Ref = if ($env:PROPHET_REF) { $env:PROPHET_REF } else { 'main' } }
    if (-not $Repo) { $Repo = if ($env:PROPHET_REPO) { $env:PROPHET_REPO } else { 'speed25200-cyber/LLM-and-Classifier' } }
    if (-not $Source) { $Source = $env:PROPHET_SOURCE }
    if (-not $DataDir) { $DataDir = $env:PROPHET_HOME }
    if ($env:PROPHET_NO_SHORTCUT -eq '1') { $NoShortcut = $true }
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue' # la barre de progression rend Invoke-WebRequest tres lent sous 5.1
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    Test-System
    $data = if ($DataDir) { $DataDir } else { Get-DefaultDataDir }
    New-Item -ItemType Directory -Path $data -Force | Out-Null
    Write-Info "dossier de donnees : $data"
    Test-Gpu
    $uv = Get-Uv

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ('prophet-studio-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    try {
        $stage = Get-Sources -Tmp $tmp -Source $Source -Repo $Repo -Ref $Ref
        Test-RunningInstance $data
        $webZip = Copy-App -Stage $stage -Data $data -Tmp $tmp -Repo $Repo
        Sync-Environment $uv $data
        if ($webZip) {
            Expand-Archive -LiteralPath $webZip -DestinationPath (Join-Path $data 'app\prophet_studio\web') -Force
            Write-Ok 'interface extraite'
        }
    } finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }

    $cli = Get-VenvTool $data 'prophet-studio'
    if (-not $cli) { throw "prophet-studio introuvable dans $(Join-Path $data 'app-venv')" }
    Invoke-Native -FilePath $cli -Arguments @('--help') -What 'prophet-studio --help' -Quiet
    Write-Ok "prophet-studio pret : $cli"

    $launcher = Write-PywLauncher $data
    if (-not $NoShortcut) { New-Shortcuts $data $launcher }

    Write-Step 'Prophet Studio est installe'
    Write-Info 'Lancer   : raccourci "Prophet Studio" (menu Demarrer ou Bureau) ; aucune console ne s''ouvre,'
    Write-Info '           l''interface s''ouvre dans votre navigateur sur http://127.0.0.1:7878'
    Write-Info "           en ligne de commande : & '$cli'"
    Write-Info 'Ensuite  : l''assistant de l''interface detecte la carte graphique et telecharge'
    Write-Info '           Bonsai 2 27B (~6 Go) et le classifieur ; comptez ~10 Go libres au total.'
    Write-Info "Journal  : $(Join-Path $data 'logs\launcher.log')"
    Write-Info 'Mettre a jour : relancez cette commande.'
    Write-Info "Desinstaller  : supprimez $data (modeles compris) et les deux raccourcis."
}

# PROPHET_INSTALLER_NO_RUN=1 : charge seulement les fonctions (tests).
if ($env:PROPHET_INSTALLER_NO_RUN -ne '1') {
    try {
        Install-ProphetStudio -Ref $Ref -Source $Source -Repo $Repo -DataDir $DataDir -NoShortcut ([bool]$NoShortcut)
    } catch {
        Write-Host ''
        Write-Host "ERREUR : $($_.Exception.Message)" -ForegroundColor Red
        Write-Host 'Relancez la commande ; si le probleme persiste, ouvrez un ticket sur GitHub avec ce message.' -ForegroundColor Red
        # pas de `exit` lorsque le script est execute par `irm | iex` : cela fermerait la fenetre
        if ($PSCommandPath) { exit 1 }
    }
}

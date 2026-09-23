# Windows PowerShell : equivalent de scripts/start_bonsai.sh pour le profil RTX 4060 8 Go (qualite).
# Prerequis : binaires CUDA du fork PrismML (https://github.com/PrismML-Eng/llama.cpp/releases, asset win-cuda-12.4-x64)
# dans .\bin\cuda, poids dans .\models (voir docs/00-PROTOCOLE-FUSION-JEV-BONSAI.md, phase 1).
# -Ctx = contexte PAR slot : llama-server recoit -c Ctx x Slots.
param(
  [string]$Model = (Get-ChildItem .\models\bonsai2-27B\*-PTQ1_0.gguf | Select-Object -First 1).FullName,
  [string]$MMProj = (Get-ChildItem .\models\bonsai2-27B\*mmproj-Q8_0.gguf -ErrorAction SilentlyContinue | Select-Object -First 1).FullName,
  [int]$Ctx = 8192, [int]$Ngl = 99, [int]$Slots = 1, [int]$Budget = 2048, [int]$Port = 8080
)
$mm = if ($MMProj) { @("--mmproj", $MMProj, "--no-mmproj-offload") } else { @("--no-mmproj") }
& .\bin\cuda\llama-server.exe -m $Model --host 127.0.0.1 --port $Port -ngl $Ngl -fa on -c ($Ctx * $Slots) -np $Slots `
  --temp 1.0 --top-p 0.95 --top-k 20 --jinja @mm --cache-type-k q4_0 --cache-type-v q4_0 `
  --reasoning-budget $Budget --cache-ram 2048 --ctx-checkpoints 8

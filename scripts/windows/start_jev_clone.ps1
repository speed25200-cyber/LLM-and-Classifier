# Windows PowerShell : equivalent de scripts/start_jev_clone.sh (petit modele System One, 4 slots).
# -Ctx = contexte PAR slot (un etat de Prophet tient en ~2-8 k tokens) : llama-server recoit -c Ctx x Slots.
param(
  [string]$Model = (Get-ChildItem .\models\ternary-1.7B\*-Q2_0_g64.gguf | Select-Object -First 1).FullName,
  [int]$Ctx = 8192, [int]$Ngl = 99, [int]$Slots = 4, [int]$Port = 8081
)
& .\bin\cuda\llama-server.exe -m $Model --host 127.0.0.1 --port $Port -ngl $Ngl -fa on -c ($Ctx * $Slots) -np $Slots `
  -b 2048 -ub 512 --cache-ram 1024 --ctx-checkpoints 8 --reasoning-budget 0 --no-mmproj

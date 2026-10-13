$ErrorActionPreference = "Stop"

$pythonExecutable = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$pipeline = Join-Path $PSScriptRoot "mammo_pipeline.py"
$seeds = @(42, 123, 2026)

foreach ($experimentSeed in $seeds) {
    $runId = "full_efficientnet384_seed${experimentSeed}_20260907"
    Write-Output "[$(Get-Date -Format s)] Iniciando $runId"
    & $pythonExecutable $pipeline benchmark `
        --models efficientnet_b0 `
        --epochs 5 `
        --batch-size 8 `
        --workers 0 `
        --device cuda `
        --seed $experimentSeed `
        --image-size 384 `
        --threshold-policy youden `
        --bootstrap-iterations 1000 `
        --learning-rate 2e-5 `
        --weight-decay 1e-4 `
        --patience 2 `
        --min-delta 1e-4 `
        --gradient-clip 1.0 `
        --mixed-precision `
        --run-id $runId `
        --output-root artifacts\benchmarks
    if ($LASTEXITCODE -ne 0) {
        throw "A execução $runId terminou com código $LASTEXITCODE."
    }
    Write-Output "[$(Get-Date -Format s)] Concluído $runId"
}

Write-Output "[$(Get-Date -Format s)] Todas as seeds foram concluídas."

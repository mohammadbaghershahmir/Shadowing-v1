# Wait for CSN-V3 training to finish, then build Dataset_V4 and start CSN-V4 training.
# Retries build/train on failure (up to 3 attempts).

$ErrorActionPreference = "Continue"
$Root = "E:\Shadowing\code"
$Python = "C:\Users\Allah\Gigazaki\Scripts\python.exe"
$LogDir = Join-Path $Root "runs"
$LogFile = Join-Path $LogDir "csn_v4_autostart.log"
$DatasetRoot = "E:\Shadowing\Dataset_V4"
$SourceDataset = "E:\Shadowing\Dataset_V3"
$MaxAttempts = 3

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

function Write-Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Add-Content -Path $LogFile -Value $line
    Write-Host $line
}

function Test-V3TrainingRunning {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "train_csn_v3\.py" } |
        ForEach-Object { return $true }
    return $false
}

function Test-BuildOk {
    $summary = Join-Path $DatasetRoot "manifests\build_summary.json"
    if (-not (Test-Path $summary)) { return $false }
    try {
        $j = Get-Content $summary -Raw | ConvertFrom-Json
        return ($j.scenes -gt 0 -and $j.tiles -gt 0)
    } catch { return $false }
}

Write-Log "=== CSN-V4 autostart watcher (retry=$MaxAttempts) ==="
Write-Log "Waiting for train_csn_v3.py to exit..."

while (Test-V3TrainingRunning) {
    Start-Sleep -Seconds 60
}

Write-Log "V3 finished. Waiting 45s for GPU release..."
Start-Sleep -Seconds 45
Set-Location $Root

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    Write-Log "--- Attempt $attempt / $MaxAttempts ---"

    Write-Log "Building Dataset_V4..."
    & $Python tools/build_dataset_v4.py --dataset-root $DatasetRoot --source-dataset $SourceDataset 2>&1 |
        Tee-Object -FilePath $LogFile -Append
    if (-not (Test-BuildOk)) {
        Write-Log "ERROR: Dataset build failed or empty (scenes=0). Retrying in 30s..."
        Start-Sleep -Seconds 30
        continue
    }
    Write-Log "Dataset_V4 OK."

    Write-Log "Starting CSN-V4 training..."
    & $Python scripts/train_csn_v4.py `
        --config configs/csn_v4_bw_overfit_full_crop.yaml `
        --device cuda `
        --skip-audit 2>&1 | Tee-Object -FilePath $LogFile -Append

    if ($LASTEXITCODE -eq 0) {
        Write-Log "CSN-V4 training completed successfully."
        exit 0
    }
    Write-Log "ERROR: train_csn_v4 exited with code $LASTEXITCODE. Retrying in 60s..."
    Start-Sleep -Seconds 60
}

Write-Log "FATAL: All $MaxAttempts attempts failed. Check log and fix manually."
exit 1

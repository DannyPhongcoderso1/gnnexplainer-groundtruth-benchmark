# Dot-source this file. It defines commands and does not start training.
$Task28Root = 'D:\CS\KLTN\gnnexplainer-groundtruth-benchmark'

function Assert-Task28Freeze {
    Set-Location -LiteralPath $Task28Root
    $env:ELLIPTIC_BASELINE_REPO = 'D:\CS\KLTN\elliptic-gnn-xai'
    $planPath = 'docs/elliptic_pgexplainer_step04_compute_plan.json'
    $plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
    $base = $plan.implementation_commit
    $commits = @(git log --first-parent --reverse --format=%H "$base..HEAD" -- $planPath)
    if ($LASTEXITCODE -ne 0 -or $commits.Count -eq 0) { throw 'Plan commit not found' }
    $frozen = $commits[0].Trim()
    $head = (git rev-parse HEAD).Trim()
    $parent = (git rev-parse "$frozen^1").Trim()
    if ($head -ne $frozen -or $parent -ne $base) { throw 'HEAD differs from frozen execution commit' }
    git diff --quiet HEAD -- docs/elliptic_pgexplainer_step04_compute_plan.json docs/elliptic_pgexplainer_step04_runbook.ps1
    if ($LASTEXITCODE -ne 0) { throw 'Plan/runbook has uncommitted changes' }
    & nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv
    if ($LASTEXITCODE -ne 0) { throw 'NVIDIA GPU visibility check failed' }
    $check = @'
import hashlib,json,shutil,sys
from pathlib import Path
import torch
import scripts.run_elliptic_pgexplainer_training as r
p=json.loads(Path('docs/elliptic_pgexplainer_step04_compute_plan.json').read_text())
assert torch.cuda.is_available(), 'CUDA unavailable'
assert shutil.disk_usage('.').free >= p['disk_estimate']['required_free_bytes'], 'Insufficient disk reserve'
for path,digest in p['file_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, 'File hash mismatch: '+path
actual=r.hard_compatibility(torch.device('cuda'))
assert actual.pop('git_commit')==sys.argv[1], 'Git identity mismatch'
assert actual==p['hard_compatibility_without_git'], 'Source/environment identity mismatch'
print('Frozen preflight PASS, execution HEAD='+sys.argv[1])
'@
    & .\.venv\Scripts\python.exe -c $check $frozen
    if ($LASTEXITCODE -ne 0) { throw 'Frozen hash/environment preflight failed' }
    return $frozen
}

function Invoke-FrozenRun {
    param(
        [Parameter(Mandatory=$true)][ValidateSet('gcn','graphsage')][string]$Backbone,
        [Parameter(Mandatory=$true)][ValidateSet(42,0,1,2,3,4)][int]$Seed,
        [switch]$ValidateOnly
    )
    Set-Location -LiteralPath $Task28Root
    # One operator process on this Windows machine; never launch a second GPU job.
    $mutex = New-Object System.Threading.Mutex($false, 'Local\Task28EllipticPGTraining')
    $acquired = $false
    try {
        try { $acquired = $mutex.WaitOne(0) }
        catch [System.Threading.AbandonedMutexException] { $acquired = $true }
        if (-not $acquired) { throw 'Another Task28 runbook process holds the GPU lock' }
        $null = Assert-Task28Freeze
        $plan = Get-Content docs/elliptic_pgexplainer_step04_compute_plan.json -Raw | ConvertFrom-Json
        if (-not $ValidateOnly) {
            if ($Seed -ne 42) { throw 'Stability training is NOT AUTHORIZED. Obtain a reviewed authorization before running these commands.' }
            if (-not $plan.primary_training_authorized) { throw 'Primary training is not authorized' }
        }
        $run = @($plan.runs | Where-Object { $_.backbone -eq $Backbone -and $_.seed -eq $Seed })
        if ($run.Count -ne 1) { throw 'Run identity must be unique' }
        $run = $run[0]
        $cli = @($run.argv)
        if ($ValidateOnly) {
            if (-not (Test-Path -LiteralPath $run.complete -PathType Leaf)) { throw 'No COMPLETE marker: validation refused, no training started' }
            $cli += '--dry-run'
        }
        $stem = "elliptic_pg_${Backbone}_seed${Seed}"
        $logDir = 'results/metrics/elliptic/task28_operator'
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        $start = (Get-Date).ToUniversalTime().ToString('o')
        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & .\.venv\Scripts\python.exe -u @cli 2>&1 | Tee-Object -FilePath "$logDir/$stem.console.log" -Append
            $code = $LASTEXITCODE
        } finally { $ErrorActionPreference = $previousPreference }
        [ordered]@{start_utc=$start; end_utc=(Get-Date).ToUniversalTime().ToString('o'); backbone=$Backbone; seed=$Seed; validation_only=[bool]$ValidateOnly; exit_code=$code} |
            ConvertTo-Json -Compress | Add-Content -LiteralPath "$logDir/$stem.exits.jsonl"
        if ($code -ne 0) { throw "Runner failed with exit code $code. Stop and inspect; do not reset or overwrite." }
    } finally {
        if ($acquired) { $mutex.ReleaseMutex() }
        $mutex.Dispose()
    }
}

# Primary commands: run manually, one after the other, only after reviewing the report.
# Invoke-FrozenRun -Backbone gcn -Seed 42
# Invoke-FrozenRun -Backbone graphsage -Seed 42

# Future stability commands: generated for completeness, NOT AUTHORIZED and blocked above.
# Invoke-FrozenRun -Backbone gcn -Seed 0
# Invoke-FrozenRun -Backbone graphsage -Seed 0
# Invoke-FrozenRun -Backbone gcn -Seed 1
# Invoke-FrozenRun -Backbone graphsage -Seed 1
# Invoke-FrozenRun -Backbone gcn -Seed 2
# Invoke-FrozenRun -Backbone graphsage -Seed 2
# Invoke-FrozenRun -Backbone gcn -Seed 3
# Invoke-FrozenRun -Backbone graphsage -Seed 3
# Invoke-FrozenRun -Backbone gcn -Seed 4
# Invoke-FrozenRun -Backbone graphsage -Seed 4

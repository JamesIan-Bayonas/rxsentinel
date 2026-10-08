param(
    [switch]$BackendOnly
)

$ErrorActionPreference = 'Stop'
$taskWorkspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskRuntimeRoot = Join-Path $taskWorkspace '.runtime\python'
$taskPython = Join-Path $taskRuntimeRoot 'cpython-3.12.12-windows-x86_64-none\python.exe'
$taskEnvironment = Join-Path $taskWorkspace '.venv'
$taskBackupRoot = Join-Path $taskWorkspace '.runtime\environment-backups'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. Install uv, open a new PowerShell terminal, and run this script again.'
}

Push-Location -LiteralPath $taskWorkspace
try {
    if (-not (Test-Path -LiteralPath $taskPython)) {
        & uv python install 3.12.12 --install-dir $taskRuntimeRoot --no-bin --no-registry
        # Some Windows uv releases fail while creating a minor-version junction after
        # publishing the actual interpreter. Only a working, exact interpreter is accepted.
        if (-not (Test-Path -LiteralPath $taskPython)) {
            throw 'The project Python runtime could not be installed.'
        }
    }
    $taskVersion = & $taskPython --version
    if ($LASTEXITCODE -ne 0 -or $taskVersion -ne 'Python 3.12.12') {
        throw 'The project runtime failed its Python 3.12.12 check.'
    }

    $taskConfig = Join-Path $taskEnvironment 'pyvenv.cfg'
    $taskExpectedHome = Split-Path -Parent $taskPython
    $taskAlreadyLocal = (Test-Path -LiteralPath $taskConfig) -and
        ((Get-Content -LiteralPath $taskConfig) -contains "home = $taskExpectedHome")
    $taskBackup = $null
    if (-not $taskAlreadyLocal -and (Test-Path -LiteralPath $taskEnvironment)) {
        # Preserve the existing environment. Both resolved paths must stay in this checkout.
        $taskSource = (Resolve-Path -LiteralPath $taskEnvironment).Path
        $taskWorkspacePrefix = $taskWorkspace.TrimEnd('\') + '\'
        if (-not $taskSource.StartsWith($taskWorkspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Environment path resolves outside the project; refusing to move it.'
        }
        New-Item -ItemType Directory -Path $taskBackupRoot -Force | Out-Null
        $taskBackup = Join-Path $taskBackupRoot ('venv-' + [guid]::NewGuid().ToString('N'))
        if (-not [IO.Path]::GetFullPath($taskBackup).StartsWith($taskWorkspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Backup path resolves outside the project; refusing to move the environment.'
        }
        Move-Item -LiteralPath $taskSource -Destination $taskBackup
        Write-Output "Previous environment preserved at $taskBackup"
    }

    if (-not $taskAlreadyLocal) {
        & uv venv --python $taskPython $taskEnvironment
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the project virtual environment.' }
    }
    $taskSyncArguments = @('sync', '--locked', '--python', $taskPython)
    if (-not $BackendOnly) { $taskSyncArguments += @('--extra', 'vision') }
    & uv @taskSyncArguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Locked dependency sync failed. Any previous environment is retained in .runtime/environment-backups.'
    }
    Write-Output 'Verifying Python version and base runtime:'
    & (Join-Path $taskEnvironment 'Scripts\python.exe') -c 'import sys; print(sys.version.split()[0]); print(sys.base_prefix)'
    if ($LASTEXITCODE -ne 0) { throw 'The rebuilt virtual environment failed verification.' }
    Write-Output 'Use uv run --locked --extra vision rxsentinel-db status in a fresh terminal.'
} finally {
    Pop-Location
}

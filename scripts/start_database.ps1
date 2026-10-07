param(
    [string]$XamppDirectory = 'C:\xampp',
    [int]$Port = 3307
)

$ErrorActionPreference = 'Stop'
$taskWorkspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskDataDirectory = Join-Path $taskWorkspace 'data\mariadb'
$taskDatabaseExecutable = Join-Path $XamppDirectory 'mysql\bin\mysqld.exe'
$taskInstaller = Join-Path $XamppDirectory 'mysql\bin\mysql_install_db.exe'
if (-not (Test-Path -LiteralPath $taskDatabaseExecutable)) {
    throw 'MariaDB executable was not found in the supplied XAMPP directory.'
}

if (-not (Test-Path -LiteralPath (Join-Path $taskDataDirectory 'mysql'))) {
    & $taskInstaller "--datadir=$taskDataDirectory" "--port=$Port"
    if ($LASTEXITCODE -ne 0) { throw 'Project database initialization failed.' }
}

$taskConfig = Join-Path $taskDataDirectory 'my.ini'
# Separate project data files and a loopback-only listener. No existing XAMPP config is edited.
# Native asynchronous I/O stalled on this Windows/XAMPP runtime during catalog writes.
# Keep the workaround local to this project instance; do not change XAMPP's global settings.
& $taskDatabaseExecutable "--defaults-file=$taskConfig" "--port=$Port" '--bind-address=127.0.0.1' '--innodb-use-native-aio=0' --console
if ($LASTEXITCODE -ne 0) { throw 'Project database stopped unexpectedly.' }

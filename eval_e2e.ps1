param(
    [string]$ApiKey,
    [string]$ScriptPath = '.\eval_e2e.py',
    [switch]$HelpOnly,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$scriptDir = Split-Path -Parent $PSCommandPath

$launcherCommonPath = Join-Path $scriptDir 'launcher-common.ps1'
if (-not (Test-Path -LiteralPath $launcherCommonPath)) {
    Write-Error "Missing helper script: $launcherCommonPath"
    exit 1
}
. $launcherCommonPath

$resolvedScriptPath = Resolve-PathValue -PathValue $ScriptPath -BaseDir $scriptDir
if (-not (Test-Path -LiteralPath $resolvedScriptPath)) {
    Write-Error "Python script not found: $resolvedScriptPath"
    exit 1
}

if (-not $HelpOnly -and [string]::IsNullOrWhiteSpace($ApiKey)) {
    $ApiKey = $env:AZURE_OPENAI_API_KEY
}

if (-not $HelpOnly -and [string]::IsNullOrWhiteSpace($ApiKey)) {
    Write-Error "ApiKey is required. Set AZURE_OPENAI_API_KEY or pass -ApiKey <key>."
    exit 1
}

$pythonRuntime = Resolve-PythonCommandWithModule -BaseDir $scriptDir -ModuleName 'openai' -RuntimeLabel 'AOAI'
if (-not $pythonRuntime) {
    exit 1
}

$pythonCommand = $pythonRuntime.Command
$pythonCommandArgs = $pythonRuntime.Args
$pythonCommandDisplay = $pythonRuntime.Display

if (-not [string]::IsNullOrWhiteSpace($ApiKey)) {
    $env:AZURE_OPENAI_API_KEY = $ApiKey
}

Write-Host "Launching $resolvedScriptPath with $pythonCommandDisplay" -ForegroundColor Cyan

$effectiveArgs = @($resolvedScriptPath)
if ($HelpOnly) {
    $effectiveArgs += '--help'
}
elseif ($ScriptArgs) {
    $effectiveArgs += $ScriptArgs
}

& $pythonCommand @pythonCommandArgs @effectiveArgs
exit $LASTEXITCODE

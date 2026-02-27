param(
    [string]$Model = 'gpt-5.2',
    [string]$OrginalTransFile = '.\sample_multi_input.txt',
    [string]$OutputFile = 'eval',
    [string]$PatchResultFile,
    [string]$PromptFile = '.\prompt_eval.txt',
    [double]$Timeout = 600,
    [int]$TimeoutRetries = 2,
    [int]$EmptyResultRetries = 2,
    [int]$ModelMismatchRetries = 2,
    [int]$Concurrency = 10,
    [string]$ScriptPath = '.\eval_copilot.py',
    [switch]$HelpOnly
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

$resolvedInputPath = Resolve-PathValue -PathValue $OrginalTransFile -BaseDir $scriptDir
$resolvedOutputPath = Resolve-PathValue -PathValue $OutputFile -BaseDir $scriptDir
$resolvedPromptFilePath = Resolve-PathValue -PathValue $PromptFile -BaseDir $scriptDir
$resolvedPatchResultPath = if ([string]::IsNullOrWhiteSpace($PatchResultFile)) {
    ''
}
else {
    ($PatchResultFile -split ',' | ForEach-Object {
        $item = $_.Trim()
        if (-not [string]::IsNullOrWhiteSpace($item)) {
            Resolve-PathValue -PathValue $item -BaseDir $scriptDir
        }
    }) -join ','
}

if (-not $HelpOnly) {
    if (-not (Test-Path -LiteralPath $resolvedInputPath)) {
        Write-Error "Input file not found: $resolvedInputPath"
        exit 1
    }
    if (-not (Test-Path -LiteralPath $resolvedPromptFilePath)) {
        Write-Error "Prompt file not found: $resolvedPromptFilePath"
        exit 1
    }
    if ([string]::IsNullOrWhiteSpace($resolvedPatchResultPath)) {
        Write-Error "PatchResultFile is required. Pass -PatchResultFile <path>."
        exit 1
    }
}

if ($HelpOnly) {
    $pythonRuntime = [PSCustomObject]@{
        Command = Join-Path $scriptDir '.venv\Scripts\python.exe'
        Args = @()
        Display = Join-Path $scriptDir '.venv\Scripts\python.exe'
    }
}
else {
    $pythonRuntime = Resolve-PythonCommandWithModule -BaseDir $scriptDir -ModuleName 'copilot' -RuntimeLabel 'Copilot evaluator'
    if (-not $pythonRuntime) {
        exit 1
    }
}

$pythonCommand = $pythonRuntime.Command
$pythonCommandArgs = $pythonRuntime.Args
$pythonCommandDisplay = $pythonRuntime.Display

Write-Host "Launching $resolvedScriptPath with $pythonCommandDisplay" -ForegroundColor Cyan
if (-not $HelpOnly) {
    Write-Host "Input file: $resolvedInputPath" -ForegroundColor Cyan
    Write-Host "Output file: $resolvedOutputPath" -ForegroundColor Cyan
    Write-Host "Patch result file: $resolvedPatchResultPath" -ForegroundColor Cyan
    Write-Host "Prompt file: $resolvedPromptFilePath" -ForegroundColor Cyan
    Write-Host "Model: $Model" -ForegroundColor Cyan
    Write-Host "Concurrency: $Concurrency" -ForegroundColor Cyan
    Write-Host "Timeout: $Timeout, retries: $TimeoutRetries" -ForegroundColor Cyan
    Write-Host "Empty-result retries: $EmptyResultRetries" -ForegroundColor Cyan
    Write-Host "Model-mismatch retries: $ModelMismatchRetries" -ForegroundColor Cyan
}

if ($HelpOnly) {
    & $pythonCommand @pythonCommandArgs $resolvedScriptPath --help
}
else {
    $scriptArgs = @(
        $resolvedScriptPath,
        '--orginal-trans-file', $resolvedInputPath,
        '--output-file', $resolvedOutputPath,
        '--prompt-file', $resolvedPromptFilePath,
        '--model', $Model,
        '--concurrency', $Concurrency,
        '--timeout', $Timeout,
        '--timeout-retries', $TimeoutRetries,
        '--empty-result-retries', $EmptyResultRetries,
        '--model-mismatch-retries', $ModelMismatchRetries,
        '--patch-result-file', $resolvedPatchResultPath
    )

    & $pythonCommand @pythonCommandArgs @scriptArgs
}

exit $LASTEXITCODE

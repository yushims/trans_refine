param(
    [string]$Deployment = 'gpt-5-chat',
    [string]$Endpoint = 'https://adaptationdev-resource.openai.azure.com/',
    [string]$ApiVersion = '2025-01-01-preview',
    [string]$ApiKey,
    [string]$OrginalTransFile = '.\sample_multi_input.txt',
    [string]$OutputFile = 'eval',
    [string]$PromptFile = '.\prompt_eval.txt',
    [string]$RepairPromptFile = '.\prompt_repair.txt',
    [string]$PatchResultFile,
    [double]$Timeout = 600,
    [int]$TimeoutRetries = 2,
    [int]$EmptyResultRetries = 2,
    [int]$Concurrency = 10,
    [string]$ScriptPath = '.\eval_aoai.py',
    [double]$Temperature = 0,
    [double]$TopP = 1,
    [double]$RetryTemperatureJitter = 0.08,
    [double]$RetryTopPJitter = 0.03,
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
$resolvedPromptPath = Resolve-PathValue -PathValue $PromptFile -BaseDir $scriptDir
$resolvedRepairPromptPath = Resolve-PathValue -PathValue $RepairPromptFile -BaseDir $scriptDir
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
    if (-not (Test-Path -LiteralPath $resolvedPromptPath)) {
        Write-Error "Prompt file not found: $resolvedPromptPath"
        exit 1
    }
    if (-not (Test-Path -LiteralPath $resolvedRepairPromptPath)) {
        Write-Error "Repair prompt file not found: $resolvedRepairPromptPath"
        exit 1
    }
    if ([string]::IsNullOrWhiteSpace($resolvedPatchResultPath)) {
        Write-Error "PatchResultFile is required. Pass -PatchResultFile <path>."
        exit 1
    }
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

Write-Host "Launching $resolvedScriptPath with $pythonCommandDisplay" -ForegroundColor Cyan
if (-not $HelpOnly) {
    Write-Host "Input file: $resolvedInputPath" -ForegroundColor Cyan
    Write-Host "Output file: $resolvedOutputPath" -ForegroundColor Cyan
    Write-Host "Patch result file: $resolvedPatchResultPath" -ForegroundColor Cyan
    Write-Host "Prompt file: $resolvedPromptPath" -ForegroundColor Cyan
    Write-Host "Repair prompt file: $resolvedRepairPromptPath" -ForegroundColor Cyan
    Write-Host "Concurrency: $Concurrency" -ForegroundColor Cyan
    Write-Host "Timeout: $Timeout, retries: $TimeoutRetries" -ForegroundColor Cyan
}

if ($HelpOnly) {
    & $pythonCommand @pythonCommandArgs $resolvedScriptPath --help
}
else {
    $scriptArgs = @(
        $resolvedScriptPath,
        '--orginal-trans-file', $resolvedInputPath,
        '--output-file', $resolvedOutputPath,
        '--prompt-file', $resolvedPromptPath,
        '--repair-prompt-file', $resolvedRepairPromptPath,
        '--deployment', $Deployment,
        '--endpoint', $Endpoint,
        '--api-version', $ApiVersion,
        '--api-key', $ApiKey,
        '--concurrency', $Concurrency,
        '--timeout', $Timeout,
        '--timeout-retries', $TimeoutRetries,
        '--empty-result-retries', $EmptyResultRetries,
        '--temperature', $Temperature,
        '--top-p', $TopP,
        '--retry-temperature-jitter', $RetryTemperatureJitter,
        '--retry-top-p-jitter', $RetryTopPJitter,
        '--patch-result-file', $resolvedPatchResultPath
    )

    & $pythonCommand @pythonCommandArgs @scriptArgs
}

exit $LASTEXITCODE

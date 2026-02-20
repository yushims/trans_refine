param(
    [string]$Deployment = 'gpt-5-chat',
    [string]$Endpoint = 'https://adaptationdev-resource.openai.azure.com/',
    [string]$ApiVersion = '2025-01-01-preview',
    [string]$InputFile = '.\sample_multi_input.txt',
    [string]$OutputFile = '.\sample_multi_output_aoai.json',
    [string]$OutputPlainFile = '.\sample_multi_output_aoai.txt',
    [string]$PromptFile = '.\prompt_patch_aoai.txt',
    [string]$RepairPromptFile = '.\prompt_repair_aoai.txt',
    [double]$Temperature = 0,
    [double]$TopP = 1,
    [int]$TimeoutRetries = 2,
    [ValidateSet('0', '1')]
    [string]$StrictJsonRepair = '1',
    [ValidateSet('0', '1')]
    [string]$ValidateOutput = '1',
    [string]$ScriptPath = '.\run_aoai.py',
    [switch]$Deterministic,
    [switch]$HelpOnly
)

if ($Deterministic) {
    if (-not $PSBoundParameters.ContainsKey('Temperature')) {
        $Temperature = 0
    }
    if (-not $PSBoundParameters.ContainsKey('TopP')) {
        $TopP = 1
    }

    Write-Host 'Deterministic mode enabled (Temperature=0, TopP=1 unless overridden).' -ForegroundColor Cyan
}

$scriptDir = Split-Path -Parent $PSCommandPath

$resolvedScriptPath = $ScriptPath
if (-not [System.IO.Path]::IsPathRooted($resolvedScriptPath)) {
    $resolvedScriptPath = Join-Path $scriptDir $resolvedScriptPath
}
if (-not (Test-Path -LiteralPath $resolvedScriptPath)) {
    Write-Error "Python script not found: $resolvedScriptPath"
    exit 1
}

$resolvedInputPath = $InputFile
if (-not [System.IO.Path]::IsPathRooted($resolvedInputPath)) {
    $resolvedInputPath = Join-Path $scriptDir $resolvedInputPath
}

$resolvedOutputPath = $OutputFile
if (-not [System.IO.Path]::IsPathRooted($resolvedOutputPath)) {
    $resolvedOutputPath = Join-Path $scriptDir $resolvedOutputPath
}

$resolvedOutputPlainPath = ''
if (-not [string]::IsNullOrWhiteSpace($OutputPlainFile)) {
    $resolvedOutputPlainPath = $OutputPlainFile
    if (-not [System.IO.Path]::IsPathRooted($resolvedOutputPlainPath)) {
        $resolvedOutputPlainPath = Join-Path $scriptDir $resolvedOutputPlainPath
    }
}

$resolvedPromptPath = $PromptFile
if (-not [System.IO.Path]::IsPathRooted($resolvedPromptPath)) {
    $resolvedPromptPath = Join-Path $scriptDir $resolvedPromptPath
}

$resolvedRepairPromptPath = $RepairPromptFile
if (-not [System.IO.Path]::IsPathRooted($resolvedRepairPromptPath)) {
    $resolvedRepairPromptPath = Join-Path $scriptDir $resolvedRepairPromptPath
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
}

$env:DEPLOYMENT_NAME = $Deployment
$env:ENDPOINT_URL = $Endpoint
$env:AZURE_OPENAI_API_VERSION = $ApiVersion
$env:AOAI_TEMPERATURE = [string]$Temperature
$env:AOAI_TOP_P = [string]$TopP
$env:TOP_P = [string]$TopP
$env:AOAI_TIMEOUT_RETRIES = [string]$TimeoutRetries
$env:STRICT_JSON_REPAIR = $StrictJsonRepair
$env:AOAI_VALIDATE_OUTPUT = $ValidateOutput
$env:PROMPT_FILE = $resolvedPromptPath
$env:AOAI_REPAIR_PROMPT_FILE = $resolvedRepairPromptPath

$pythonCommand = $null
$pythonCommandArgs = @()

$workspaceVenvPython = Join-Path $scriptDir '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $workspaceVenvPython) {
    $pythonCommand = $workspaceVenvPython
}

if (-not $pythonCommand -and $env:VIRTUAL_ENV) {
    $activeVenvPython = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
    if (Test-Path -LiteralPath $activeVenvPython) {
        $pythonCommand = $activeVenvPython
    }
}

if (-not $pythonCommand -and $env:CONDA_PREFIX) {
    $activeCondaPython = Join-Path $env:CONDA_PREFIX 'python.exe'
    if (Test-Path -LiteralPath $activeCondaPython) {
        $pythonCommand = $activeCondaPython
    }
}

if (-not $pythonCommand) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $pythonCommand = $pythonCmd.Source
    }
}

if (-not $pythonCommand) {
    $python313Cmd = Get-Command python3.13 -ErrorAction SilentlyContinue
    if ($python313Cmd) {
        $pythonCommand = $python313Cmd.Source
    }
}

if (-not $pythonCommand) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $pythonCommand = $pyLauncher.Source
        $pythonCommandArgs = @('-3.13')
    }
}

if (-not $pythonCommand) {
    Write-Error 'No Python executable found (.venv, active env, python, python3.13, or py -3.13).'
    exit 1
}

$pythonCommandDisplay = $pythonCommand
if ($pythonCommandArgs.Count -gt 0) {
    $pythonCommandDisplay = "$pythonCommand $($pythonCommandArgs -join ' ')"
}

Write-Host "Launching $resolvedScriptPath with $pythonCommandDisplay" -ForegroundColor Cyan
if (-not $HelpOnly) {
    Write-Host "Input file: $resolvedInputPath" -ForegroundColor Cyan
    Write-Host "Output file: $resolvedOutputPath" -ForegroundColor Cyan
    if (-not [string]::IsNullOrWhiteSpace($resolvedOutputPlainPath)) {
        Write-Host "Plain output file: $resolvedOutputPlainPath" -ForegroundColor Cyan
    }
    Write-Host "Prompt file: $resolvedPromptPath" -ForegroundColor Cyan
    Write-Host "Repair prompt file: $resolvedRepairPromptPath" -ForegroundColor Cyan
}

if ($HelpOnly) {
    & $pythonCommand @pythonCommandArgs $resolvedScriptPath --help
}
else {
    if (-not [string]::IsNullOrWhiteSpace($resolvedOutputPlainPath)) {
        & $pythonCommand @pythonCommandArgs $resolvedScriptPath --input-file $resolvedInputPath --output-file $resolvedOutputPath --output-plain-file $resolvedOutputPlainPath --prompt-file $resolvedPromptPath --repair-prompt-file $resolvedRepairPromptPath --deployment $Deployment --endpoint $Endpoint --api-version $ApiVersion --validate-output $ValidateOutput
    }
    else {
        & $pythonCommand @pythonCommandArgs $resolvedScriptPath --input-file $resolvedInputPath --output-file $resolvedOutputPath --prompt-file $resolvedPromptPath --repair-prompt-file $resolvedRepairPromptPath --deployment $Deployment --endpoint $Endpoint --api-version $ApiVersion --validate-output $ValidateOutput
    }
}

exit $LASTEXITCODE

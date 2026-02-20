param(
    [string]$Model = 'gpt-5.2',
    [string]$PromptFile = 'prompt_patch_copilot.txt',
    [string]$RepairPromptFile = 'prompt_repair_copilot.txt',
    [double]$Timeout = 180,
    [int]$TimeoutRetries = 2,
    [ValidateSet('0', '1')]
    [string]$StrictJsonRepair = '1',
    [ValidateSet('0', '1')]
    [string]$ValidateOutput = '1',
    [string]$ScriptPath = '.\run_copilot.py',
    [string]$InputFile = '.\sample_multi_input.txt',
    [string]$OutputFile = '.\sample_multi_output_copilot.json',
    [string]$OutputPlainFile = '',
    [switch]$ListModelsOnly,
    [switch]$HelpOnly
)

$scriptDir = Split-Path -Parent $PSCommandPath
$envScript = Join-Path $scriptDir 'copilot-env.ps1'

if (-not (Test-Path -LiteralPath $envScript)) {
    Write-Error "Missing env script: $envScript"
    exit 1
}

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

$resolvedPromptPath = $PromptFile
if (-not [System.IO.Path]::IsPathRooted($resolvedPromptPath)) {
    $resolvedPromptPath = Join-Path $scriptDir $resolvedPromptPath
}

$resolvedRepairPromptPath = $RepairPromptFile
if (-not [System.IO.Path]::IsPathRooted($resolvedRepairPromptPath)) {
    $resolvedRepairPromptPath = Join-Path $scriptDir $resolvedRepairPromptPath
}

$resolvedOutputPlainPath = ''
if (-not [string]::IsNullOrWhiteSpace($OutputPlainFile)) {
    $resolvedOutputPlainPath = $OutputPlainFile
    if (-not [System.IO.Path]::IsPathRooted($resolvedOutputPlainPath)) {
        $resolvedOutputPlainPath = Join-Path $scriptDir $resolvedOutputPlainPath
    }
}

if (-not $HelpOnly -and -not $ListModelsOnly) {
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

& $envScript -Action set `
    -Model $Model `
    -PromptFile $resolvedPromptPath `
    -Timeout $Timeout `
    -TimeoutRetries $TimeoutRetries `
    -StrictJsonRepair $StrictJsonRepair `
    -ValidateOutput $ValidateOutput

if (-not $?) {
    exit 1
}

$pythonCommand = $null
if (Get-Command python3.13 -ErrorAction SilentlyContinue) {
    $pythonCommand = 'python3.13'
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCommand = 'python'
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCommand = 'py -3.13'
}

if (-not $pythonCommand) {
    Write-Error 'No Python executable found (python3.13, python, or py -3.13).'
    exit 1
}

Write-Host "Launching run_copilot with $pythonCommand" -ForegroundColor Cyan
if ($ListModelsOnly) {
    Write-Host "List-models-only mode enabled." -ForegroundColor Cyan
}
elseif (-not $HelpOnly) {
    Write-Host "Input file: $resolvedInputPath" -ForegroundColor Cyan
    Write-Host "Output file: $resolvedOutputPath" -ForegroundColor Cyan
    Write-Host "Prompt file: $resolvedPromptPath" -ForegroundColor Cyan
    Write-Host "Repair prompt file: $resolvedRepairPromptPath" -ForegroundColor Cyan
    if (-not [string]::IsNullOrWhiteSpace($resolvedOutputPlainPath)) {
        Write-Host "Plain output file: $resolvedOutputPlainPath" -ForegroundColor Cyan
    }
}

if ($HelpOnly) {
    if ($pythonCommand -eq 'py -3.13') {
        & py -3.13 $resolvedScriptPath --help
    }
    else {
        & $pythonCommand $resolvedScriptPath --help
    }
}
elseif ($ListModelsOnly) {
    if ($pythonCommand -eq 'py -3.13') {
        & py -3.13 $resolvedScriptPath --list-models-only --model $Model
    }
    else {
        & $pythonCommand $resolvedScriptPath --list-models-only --model $Model
    }
}
else {
    if ($pythonCommand -eq 'py -3.13') {
        if (-not [string]::IsNullOrWhiteSpace($resolvedOutputPlainPath)) {
            & py -3.13 $resolvedScriptPath --input-file $resolvedInputPath --output-file $resolvedOutputPath --output-plain-file $resolvedOutputPlainPath --prompt-file $resolvedPromptPath --repair-prompt-file $resolvedRepairPromptPath --model $Model --validate-output $ValidateOutput
        }
        else {
            & py -3.13 $resolvedScriptPath --input-file $resolvedInputPath --output-file $resolvedOutputPath --prompt-file $resolvedPromptPath --repair-prompt-file $resolvedRepairPromptPath --model $Model --validate-output $ValidateOutput
        }
    }
    else {
        if (-not [string]::IsNullOrWhiteSpace($resolvedOutputPlainPath)) {
            & $pythonCommand $resolvedScriptPath --input-file $resolvedInputPath --output-file $resolvedOutputPath --output-plain-file $resolvedOutputPlainPath --prompt-file $resolvedPromptPath --repair-prompt-file $resolvedRepairPromptPath --model $Model --validate-output $ValidateOutput
        }
        else {
            & $pythonCommand $resolvedScriptPath --input-file $resolvedInputPath --output-file $resolvedOutputPath --prompt-file $resolvedPromptPath --repair-prompt-file $resolvedRepairPromptPath --model $Model --validate-output $ValidateOutput
        }
    }
}

exit $LASTEXITCODE

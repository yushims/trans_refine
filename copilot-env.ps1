param(
    [ValidateSet('show', 'set', 'clear')]
    [string]$Action = 'show',

    [string]$Model = 'gpt-5.2-thinking',
    [string]$PromptFile = 'prompt_patch_copilot.txt',
    [double]$Timeout = 180,
    [int]$TimeoutRetries = 2,
    [ValidateSet('0', '1')]
    [string]$StrictJsonRepair = '1',
    [ValidateSet('0', '1')]
    [string]$ValidateOutput = '1'
)

$envVarNames = @(
    'COPILOT_MODEL',
    'PROMPT_FILE',
    'COPILOT_TIMEOUT',
    'COPILOT_TIMEOUT_RETRIES',
    'STRICT_JSON_REPAIR',
    'COPILOT_VALIDATE_OUTPUT'
)

function Show-Env {
    Write-Host "Current Copilot environment variables:" -ForegroundColor Cyan
    foreach ($name in $envVarNames) {
        $value = [Environment]::GetEnvironmentVariable($name, 'Process')
        if ([string]::IsNullOrEmpty($value)) {
            Write-Host "  $name = <not set>"
        }
        else {
            Write-Host "  $name = $value"
        }
    }
}

function Resolve-PromptFile([string]$PathValue) {
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $PathValue
    }

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }

    if (Test-Path -LiteralPath $PathValue) {
        return (Resolve-Path -LiteralPath $PathValue).Path
    }

    $scriptDirPath = Split-Path -Parent $PSCommandPath
    $candidate = Join-Path $scriptDirPath $PathValue
    if (Test-Path -LiteralPath $candidate) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }

    return $PathValue
}

switch ($Action) {
    'set' {
        $resolvedPromptFile = Resolve-PromptFile -PathValue $PromptFile

        $env:COPILOT_MODEL = $Model
        $env:PROMPT_FILE = $resolvedPromptFile
        $env:COPILOT_TIMEOUT = [string]$Timeout
        $env:COPILOT_TIMEOUT_RETRIES = [string]$TimeoutRetries
        $env:STRICT_JSON_REPAIR = $StrictJsonRepair
        $env:COPILOT_VALIDATE_OUTPUT = $ValidateOutput

        Write-Host "Copilot environment variables updated for this terminal session." -ForegroundColor Green
        Show-Env
    }
    'clear' {
        foreach ($name in $envVarNames) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        }

        Write-Host "Copilot environment variables cleared for this terminal session." -ForegroundColor Yellow
        Show-Env
    }
    default {
        Show-Env
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor Cyan
        Write-Host "  .\copilot-env.ps1 -Action set"
        Write-Host "  .\copilot-env.ps1 -Action set -Timeout 300 -TimeoutRetries 3"
        Write-Host "  .\copilot-env.ps1 -Action set -StrictJsonRepair 0"
        Write-Host "  .\copilot-env.ps1 -Action set -ValidateOutput 1"
        Write-Host "  .\copilot-env.ps1 -Action clear"
    }
}

param(
    [string]$Model = 'gpt-5.2',
    [string]$InputFile = '.\sample_multi_input.txt',
    [string]$OutputFile = '.\sample_multi_output_copilot',
    [string]$PatchPromptFile = '.\prompt_patch.txt',
    [string]$RepairPromptFile = '.\prompt_repair.txt',
    [double]$Timeout = 600,
    [int]$TimeoutRetries = 2,
    [int]$EmptyResultRetries = 2,
    [int]$ModelMismatchRetries = 2,
    [int]$Concurrency = 10,
    [string]$ScriptPath = '.\run_copilot.py',
    [switch]$ListModelsOnly,
    [switch]$PrintModels,
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

$resolvedInputPath = Resolve-PathValue -PathValue $InputFile -BaseDir $scriptDir
$resolvedOutputBasePath = Resolve-PathValue -PathValue $OutputFile -BaseDir $scriptDir
$outputExtension = [System.IO.Path]::GetExtension($resolvedOutputBasePath)
if ([string]::IsNullOrWhiteSpace($outputExtension)) {
    $resolvedOutputPath = "$resolvedOutputBasePath.json"
    $resolvedOutputTextPath = "$resolvedOutputBasePath.txt"
}
else {
    $normalizedOutputExtension = $outputExtension.ToLowerInvariant()
    if ($normalizedOutputExtension -eq '.json' -or $normalizedOutputExtension -eq '.txt') {
        $outputBasePath = $resolvedOutputBasePath.Substring(0, $resolvedOutputBasePath.Length - $outputExtension.Length)
    }
    else {
        $outputBasePath = $resolvedOutputBasePath
    }
    $resolvedOutputPath = "$outputBasePath.json"
    $resolvedOutputTextPath = "$outputBasePath.txt"
}
$resolvedPatchPromptPath = Resolve-PathValue -PathValue $PatchPromptFile -BaseDir $scriptDir
$resolvedRepairPromptPath = Resolve-PathValue -PathValue $RepairPromptFile -BaseDir $scriptDir

if (-not $HelpOnly -and -not $ListModelsOnly) {
    if (-not (Test-Path -LiteralPath $resolvedInputPath)) {
        Write-Error "Input file not found: $resolvedInputPath"
        exit 1
    }
    if (-not (Test-Path -LiteralPath $resolvedPatchPromptPath)) {
        Write-Error "Patch prompt file not found: $resolvedPatchPromptPath"
        exit 1
    }
    if (-not (Test-Path -LiteralPath $resolvedRepairPromptPath)) {
        Write-Error "Repair prompt file not found: $resolvedRepairPromptPath"
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
    try {
        $pythonRuntime = Resolve-PythonCommandWithModule -BaseDir $scriptDir -ModuleName 'copilot' -RuntimeLabel 'Copilot'
        $copilotClientArgs = @()
        if ($pythonRuntime.Args) {
            $copilotClientArgs += $pythonRuntime.Args
        }
        $copilotClientArgs += @('-c', 'from copilot import CopilotClient')
        $copilotClientCheck = & $pythonRuntime.Command @copilotClientArgs 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "`.venv contains a 'copilot' module but it does not expose CopilotClient (expected runtime integration missing)."
        }
    }
    catch {
        $workspaceVenvActivate = Join-Path $scriptDir '.venv\Scripts\Activate.ps1'
        $workspaceVenvPython = Join-Path $scriptDir '.venv\Scripts\python.exe'
        Write-Error $_
        Write-Host "Copilot runtime preflight (.venv only):" -ForegroundColor Yellow
        Write-Host "  1) Activate .venv:" -ForegroundColor Yellow
        Write-Host "     & `"$workspaceVenvActivate`"" -ForegroundColor Yellow
        Write-Host "  2) Install dependencies into .venv:" -ForegroundColor Yellow
        Write-Host "     & `"$workspaceVenvPython`" -m pip install -r requirements.txt" -ForegroundColor Yellow
        Write-Host "  3) Verify expected Copilot runtime symbol in .venv:" -ForegroundColor Yellow
        Write-Host ('     & "{0}" -c "from copilot import CopilotClient; print(CopilotClient)"' -f $workspaceVenvPython) -ForegroundColor Yellow
        Write-Host "  4) If step 3 fails, this machine is missing the internal Copilot runtime integration; the public PyPI 'copilot' package is not sufficient." -ForegroundColor Yellow
        exit 1
    }
}

$pythonCommand = $pythonRuntime.Command
$pythonCommandArgs = $pythonRuntime.Args
$pythonCommandDisplay = $pythonRuntime.Display

Write-Host "Launching $resolvedScriptPath with $pythonCommandDisplay" -ForegroundColor Cyan
if ($ListModelsOnly) {
    Write-Host "List-models-only mode enabled." -ForegroundColor Cyan
}
elseif ($PrintModels) {
    Write-Host "Print-models mode enabled." -ForegroundColor Cyan
}
elseif (-not $HelpOnly) {
    Write-Host "Input file: $resolvedInputPath" -ForegroundColor Cyan
    Write-Host "Output file: $resolvedOutputPath" -ForegroundColor Cyan
    Write-Host "Patch prompt file: $resolvedPatchPromptPath" -ForegroundColor Cyan
    Write-Host "Repair prompt file: $resolvedRepairPromptPath" -ForegroundColor Cyan
    Write-Host "Text output file: $resolvedOutputTextPath" -ForegroundColor Cyan
    Write-Host "Concurrency: $Concurrency" -ForegroundColor Cyan
    Write-Host "Timeout: $Timeout, retries: $TimeoutRetries" -ForegroundColor Cyan
}

if ($HelpOnly) {
    Write-Host "Tip: use -PrintModels to print client.list_models() before a normal run." -ForegroundColor DarkCyan
    & $pythonCommand @pythonCommandArgs $resolvedScriptPath --help
}
elseif ($ListModelsOnly) {
    & $pythonCommand @pythonCommandArgs $resolvedScriptPath --list-models-only --model $Model --patch-prompt-file $resolvedPatchPromptPath --repair-prompt-file $resolvedRepairPromptPath
}
else {
    $scriptArgs = @(
        $resolvedScriptPath,
        '--input-file', $resolvedInputPath,
        '--patch-prompt-file', $resolvedPatchPromptPath,
        '--repair-prompt-file', $resolvedRepairPromptPath,
        '--output-file', $resolvedOutputPath,
        '--model', $Model,
        '--concurrency', $Concurrency,
        '--timeout', $Timeout,
        '--timeout-retries', $TimeoutRetries,
        '--empty-result-retries', $EmptyResultRetries,
        '--model-mismatch-retries', $ModelMismatchRetries
    )

    if ($PrintModels) {
        $scriptArgs += '--print-models'
    }

    & $pythonCommand @pythonCommandArgs @scriptArgs
}

exit $LASTEXITCODE

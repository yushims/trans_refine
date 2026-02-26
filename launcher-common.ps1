function Resolve-PathValue {
    param(
        [string]$PathValue,
        [string]$BaseDir
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $PathValue
    }
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }

    $combined = Join-Path $BaseDir $PathValue
    return [System.IO.Path]::GetFullPath($combined)
}

function Test-PythonModuleImport {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonPath,
        [string[]]$PythonArgs = @(),
        [Parameter(Mandatory = $true)]
        [string]$ModuleName
    )

    try {
        & $PythonPath @PythonArgs -c "import $ModuleName" *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Resolve-PythonCommandWithModule {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BaseDir,
        [Parameter(Mandatory = $true)]
        [string]$ModuleName,
        [Parameter(Mandatory = $true)]
        [string]$RuntimeLabel
    )

    $workspaceVenvPython = Join-Path $BaseDir '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $workspaceVenvPython)) {
        Write-Error "Missing workspace Python environment: $workspaceVenvPython. Use .venv only."
        return $null
    }

    if (-not (Test-PythonModuleImport -PythonPath $workspaceVenvPython -ModuleName $ModuleName)) {
        Write-Error ".venv exists but missing required $RuntimeLabel runtime module '$ModuleName'. Install dependencies into .venv."
        return $null
    }

    return [PSCustomObject]@{
        Command = $workspaceVenvPython
        Args = @()
        Display = $workspaceVenvPython
    }
}

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DetbArgs
)

$ErrorActionPreference = 'Stop'

if (-not $DetbArgs -or $DetbArgs.Count -eq 0) {
    python -m detb.cli --help
    exit $LASTEXITCODE
}

python -m detb.cli @DetbArgs
exit $LASTEXITCODE

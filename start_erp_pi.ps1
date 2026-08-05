$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $projectRoot ".env"
$venvScripts = Join-Path $projectRoot ".venv\Scripts"
if (-not (Test-Path -LiteralPath (Join-Path $venvScripts "python.exe"))) {
    throw "Не найдено Python-окружение .venv. Выполните: uv venv .venv; uv pip install --python .venv\Scripts\python.exe -r requirements.txt"
}
if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Не найден .env в $projectRoot"
}

$apiKey = $null
foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
    if ($line -match '^\s*WORMSOFT_API_KEY\s*=\s*(.*)\s*$') {
        $apiKey = $Matches[1].Trim()
        if (($apiKey.StartsWith('"') -and $apiKey.EndsWith('"')) -or
            ($apiKey.StartsWith("'") -and $apiKey.EndsWith("'"))) {
            $apiKey = $apiKey.Substring(1, $apiKey.Length - 2)
        }
        break
    }
}
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "В .env отсутствует WORMSOFT_API_KEY"
}

$env:WORMSOFT_API_KEY = $apiKey
$oldPath = $env:Path
$env:Path = "$venvScripts;$oldPath"
Set-Location -LiteralPath $projectRoot
try {
    & pi --approve --model "wormsoft-gateway/wormsoft/agent/high" --name "ERP ТЗ" --append-system-prompt ".pi/prompts/coordinator.md"
} finally {
    Remove-Item Env:WORMSOFT_API_KEY -ErrorAction SilentlyContinue
    $env:Path = $oldPath
    $apiKey = $null
}

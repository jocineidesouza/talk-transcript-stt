param(
  [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"

function Resolve-InProjectPath {
  param(
    [string]$Root,
    [string]$RelativePath
  )

  $rootResolved = (Resolve-Path -LiteralPath $Root).Path
  $target = Join-Path $rootResolved $RelativePath
  return @{
    Root = $rootResolved
    Target = $target
  }
}

function Assert-PathInsideRoot {
  param(
    [string]$Root,
    [string]$Target
  )

  $fullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\')
  $fullTarget = [System.IO.Path]::GetFullPath($Target)
  if (-not $fullTarget.StartsWith($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Caminho fora da raiz do projeto: $fullTarget"
  }
}

$rootPath = (Resolve-Path -LiteralPath $ProjectRoot).Path
$composeFile = Join-Path $rootPath "docker-compose.yml"
if (-not (Test-Path -LiteralPath $composeFile)) {
  throw "docker-compose.yml nao encontrado em: $rootPath"
}

Write-Host "[1/6] Parando servico stt..."
docker compose -f $composeFile stop stt

Write-Host "[2/6] Limpando SQLite em stt/data..."
$dataPath = (Resolve-InProjectPath -Root $rootPath -RelativePath "stt/data").Target
Assert-PathInsideRoot -Root $rootPath -Target $dataPath

$sqliteFiles = @("queue.db", "queue.db-shm", "queue.db-wal")
foreach ($name in $sqliteFiles) {
  $file = Join-Path $dataPath $name
  if (Test-Path -LiteralPath $file) {
    Assert-PathInsideRoot -Root $rootPath -Target $file
    Remove-Item -LiteralPath $file -Force
    Write-Host "  removido: $file"
  }
}

Write-Host "[3/6] Limpando spool em stt/data/spool (sem apagar a pasta)..."
$spoolPath = (Resolve-InProjectPath -Root $rootPath -RelativePath "stt/data/spool").Target
Assert-PathInsideRoot -Root $rootPath -Target $spoolPath
if (Test-Path -LiteralPath $spoolPath) {
  Get-ChildItem -LiteralPath $spoolPath -Force | ForEach-Object {
    Assert-PathInsideRoot -Root $rootPath -Target $_.FullName
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
  }
} else {
  New-Item -ItemType Directory -Path $spoolPath | Out-Null
}

Write-Host "[4/6] Validando que stt/models sera preservado..."
$modelsPath = (Resolve-InProjectPath -Root $rootPath -RelativePath "stt/models").Target
Assert-PathInsideRoot -Root $rootPath -Target $modelsPath
if (-not (Test-Path -LiteralPath $modelsPath)) {
  Write-Warning "Pasta de modelos nao encontrada: $modelsPath"
}

Write-Host "[5/6] Rebuild da imagem stt..."
docker compose -f $composeFile build stt

Write-Host "[6/6] Subindo servico stt..."
docker compose -f $composeFile up -d stt
docker compose -f $composeFile ps

Write-Host "Reset concluido com sucesso."


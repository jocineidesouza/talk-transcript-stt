@echo off
setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set "PURGE_MODELS=0"
if /i "%~1"=="purge-models" set "PURGE_MODELS=1"

echo [1/5] Limpando dados locais (queue.db e spool)...
if exist "stt\data\queue.db" del /f /q "stt\data\queue.db"
if exist "stt\data\spool" (
  for /d %%D in ("stt\data\spool\*") do rmdir /s /q "%%~fD"
  del /f /q "stt\data\spool\*" 2>nul
)

if "%PURGE_MODELS%"=="1" (
  echo [2/5] Removendo stt\models (modo purge-models)...
  if exist "stt\models" rmdir /s /q "stt\models"
) else (
  echo [2/5] Mantendo stt\models para evitar novo download...
)

echo [3/5] Parando containers...
docker compose down
if errorlevel 1 goto :fail

echo [4/5] Rebuild e start...
docker compose up -d --build
if errorlevel 1 goto :fail

echo [5/5] Status atual...
docker compose ps

echo Ultimas linhas de log do servico stt...
docker compose logs --tail 40 stt

echo.
echo Concluido.
echo.
echo Uso:
echo   reset_stt.bat               ^(mantem stt\models^)
echo   reset_stt.bat purge-models  ^(remove stt\models^)
exit /b 0

:fail
echo.
echo Falha ao executar o processo. Verifique as mensagens acima.
exit /b 1

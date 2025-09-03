@echo off
setlocal enabledelayedexpansion

REM Script de atualização para TraduLibras (Python Nativo - Windows)
REM Autor: TraduLibras Team
REM Versão: 2.0.0

title TraduLibras Update

REM Banner
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                🤟 TraduLibras Update 🤟                      ║
echo ║              Sistema de Atualização Automática               ║
echo ║                        Versão 2.0.0                          ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Verificar se é um repositório Git
git status >nul 2>&1
if errorlevel 1 (
    echo [AVISO] Este não é um repositório Git.
    echo [INFO] Para atualizações automáticas, clone o projeto do GitHub:
    echo [INFO] git clone https://github.com/prof-atritiack/libras-js.git
    echo.
    echo [INFO] Ou baixe manualmente as atualizações do GitHub.
    pause
    exit /b 1
)

REM Fazer backup dos modelos
echo [INFO] Fazendo backup dos modelos...
if not exist "backup" mkdir backup

REM Backup dos modelos
if exist "modelos\modelo_libras.pkl" (
    for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set mydate=%%c%%a%%b
    for /f "tokens=1-2 delims=: " %%a in ('time /t') do set mytime=%%a%%b
    set timestamp=%mydate%_%mytime%
    copy "modelos\modelo_libras.pkl" "backup\modelo_libras_backup_%timestamp%.pkl" >nul
    echo [INFO] Backup do modelo salvo em backup\
)

REM Backup dos dados
if exist "gestos_libras.csv" (
    copy "gestos_libras.csv" "backup\gestos_libras_backup_%timestamp%.csv" >nul
    echo [INFO] Backup dos dados salvo em backup\
)

REM Verificar se há atualizações
echo [INFO] Verificando atualizações...
git fetch origin >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Erro ao buscar atualizações!
    pause
    exit /b 1
)

REM Verificar se há atualizações disponíveis
git status -uno | findstr "behind" >nul
if errorlevel 1 (
    echo [INFO] Projeto já está atualizado! ✅
    echo [INFO] Atualizando dependências...
    goto :update_deps
)

REM Aplicar atualizações
echo [INFO] Atualizações encontradas! Aplicando...
git pull origin main
if errorlevel 1 (
    echo [ERRO] Erro ao aplicar atualizações!
    echo [INFO] Verifique se há conflitos ou se o repositório está limpo.
    pause
    exit /b 1
)

echo [TraduLibras] Atualizações aplicadas com sucesso! ✅

:update_deps
REM Atualizar dependências
echo [INFO] Atualizando dependências...
python -m pip install --upgrade pip >nul 2>&1
if errorlevel 1 (
    echo [AVISO] Erro ao atualizar pip, continuando...
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERRO] Erro ao atualizar dependências!
    pause
    exit /b 1
)

echo [TraduLibras] Dependências atualizadas com sucesso! ✅

REM Perguntar sobre retreinar modelo
echo [AVISO] Deseja retreinar o modelo com os novos dados? (Y/n)
set /p response=
if not "!response!"=="n" if not "!response!"=="N" (
    echo [INFO] Retreinando modelo...
    python treinar_letras_simples.py
    if errorlevel 1 (
        echo [ERRO] Erro ao retreinar modelo!
        pause
        exit /b 1
    )
    echo [TraduLibras] Modelo retreinado com sucesso! ✅
)

echo.
echo [TraduLibras] Atualização concluída com sucesso! 🎉
echo [INFO] Execute 'python app.py' para iniciar a versão atualizada!
echo.
echo [INFO] 📱 Acesse: http://localhost:5000
echo [INFO] 🌐 Para rede local, use o botão 'Info Rede' na interface
echo.
pause

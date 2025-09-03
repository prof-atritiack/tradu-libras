@echo off
setlocal enabledelayedexpansion

REM Script para executar TraduLibras com Docker no Windows
REM Autor: TraduLibras Team
REM Versão: 2.0.0

title TraduLibras Docker

REM Banner
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    🤟 TraduLibras Docker 🤟                  ║
echo ║              Sistema de Reconhecimento de LIBRAS             ║
echo ║                        Versão 2.0.0                          ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Verificar se Docker está instalado
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Docker não está instalado!
    echo [INFO] Instale o Docker Desktop: https://docs.docker.com/desktop/windows/
    pause
    exit /b 1
)

REM Verificar se Docker Compose está instalado
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Docker Compose não está instalado!
    echo [INFO] Instale o Docker Compose: https://docs.docker.com/compose/install/
    pause
    exit /b 1
)

REM Verificar se Docker está rodando
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Docker não está rodando!
    echo [INFO] Inicie o Docker Desktop
    pause
    exit /b 1
)

echo [TraduLibras] Verificações iniciais concluídas ✅

REM Função para obter IP local
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /c:"IPv4"') do (
    set ip=%%i
    set ip=!ip: =!
    goto :found_ip
)
:found_ip

REM Processar argumentos
set "action=%1"
if "%action%"=="" set "action=start"

if "%action%"=="start" goto :start
if "%action%"=="stop" goto :stop
if "%action%"=="restart" goto :restart
if "%action%"=="build" goto :build
if "%action%"=="train" goto :train
if "%action%"=="logs" goto :logs
if "%action%"=="status" goto :status
if "%action%"=="clean" goto :clean
if "%action%"=="help" goto :help
goto :help

:start
echo [TraduLibras] Iniciando TraduLibras com Docker...

REM Verificar se precisa treinar modelo
if not exist "modelos\modelo_libras.pkl" (
    echo [AVISO] Modelo não encontrado. Deseja treinar agora? (Y/n)
    set /p response=
    if not "!response!"=="n" if not "!response!"=="N" (
        call :train_model
    )
)

echo [INFO] Construindo imagem Docker...
docker-compose build
if errorlevel 1 (
    echo [ERRO] Erro ao construir a imagem!
    pause
    exit /b 1
)

echo [INFO] Iniciando container...
docker-compose up -d
if errorlevel 1 (
    echo [ERRO] Erro ao iniciar o container!
    pause
    exit /b 1
)

echo [INFO] Aguardando container estar pronto...
timeout /t 10 /nobreak >nul

REM Verificar se está rodando
docker-compose ps | findstr "Up" >nul
if errorlevel 1 (
    echo [ERRO] Erro ao iniciar o container!
    echo [INFO] Verifique os logs: docker-compose logs
    pause
    exit /b 1
)

call :show_access_info
goto :end

:stop
echo [TraduLibras] Parando TraduLibras...
docker-compose down
echo [TraduLibras] TraduLibras parado! ✅
goto :end

:restart
echo [TraduLibras] Reiniciando TraduLibras...
docker-compose restart
call :show_access_info
goto :end

:build
echo [TraduLibras] Construindo imagem Docker...
docker-compose build --no-cache
echo [TraduLibras] Imagem construída! ✅
goto :end

:train
call :train_model
goto :end

:logs
echo [INFO] Mostrando logs do TraduLibras...
docker-compose logs -f
goto :end

:status
echo [INFO] Status do TraduLibras:
docker-compose ps
goto :end

:clean
echo [AVISO] Isso irá remover todos os containers e imagens do TraduLibras.
echo [AVISO] Deseja continuar? (y/N)
set /p response=
if "!response!"=="y" if "!response!"=="Y" (
    echo [INFO] Limpando containers e imagens...
    docker-compose down --rmi all --volumes --remove-orphans
    docker system prune -f
    echo [TraduLibras] Limpeza concluída! ✅
) else (
    echo [INFO] Operação cancelada.
)
goto :end

:help
echo [INFO] Uso: %0 [OPÇÃO]
echo.
echo Opções:
echo   start, s     Iniciar o TraduLibras (padrão)
echo   stop         Parar o TraduLibras
echo   restart      Reiniciar o TraduLibras
echo   build        Construir a imagem Docker
echo   train        Treinar o modelo de reconhecimento
echo   logs         Mostrar logs do container
echo   status       Mostrar status do container
echo   clean        Limpar containers e imagens
echo   help, h      Mostrar esta ajuda
echo.
echo Exemplos:
echo   %0 start     # Iniciar o TraduLibras
echo   %0 train     # Treinar o modelo
echo   %0 logs      # Ver logs em tempo real
goto :end

:train_model
echo [INFO] Iniciando treinamento do modelo...

if exist "modelos\modelo_libras.pkl" (
    echo [AVISO] Modelo já existe. Deseja retreinar? (y/N)
    set /p response=
    if not "!response!"=="y" if not "!response!"=="Y" (
        echo [INFO] Usando modelo existente.
        goto :eof
    )
)

echo [INFO] Executando treinamento do modelo...
docker-compose run --rm tradulibras python treinar_letras_simples.py
if errorlevel 1 (
    echo [ERRO] Erro no treinamento do modelo!
    pause
    exit /b 1
)

echo [TraduLibras] Modelo treinado com sucesso! ✅
goto :eof

:show_access_info
echo.
echo 🚀 TraduLibras está rodando!
echo ═══════════════════════════════════════════════════════════════
echo 📱 ACESSO LOCAL:
echo    http://localhost:5000
echo    http://127.0.0.1:5000
echo ═══════════════════════════════════════════════════════════════
echo 🌐 ACESSO NA REDE LOCAL:
echo    http://%ip%:5000
echo ═══════════════════════════════════════════════════════════════
echo 📱 Para acessar de outros dispositivos:
echo    1. Certifique-se que estão na mesma rede Wi-Fi
echo    2. Use o endereço: http://%ip%:5000
echo    3. Funciona em celular, tablet e outros computadores
echo ═══════════════════════════════════════════════════════════════
echo 🛠️  Comandos úteis:
echo    • Parar: docker-compose down
echo    • Logs: docker-compose logs -f
echo    • Status: docker-compose ps
echo ═══════════════════════════════════════════════════════════════
goto :eof

:end
pause

#!/usr/bin/env python3
"""
Script de configuração automática do TraduLibras
Instala dependências e configura o ambiente automaticamente
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ é necessário!")
        print(f"   Versão atual: {version.major}.{version.minor}.{version.micro}")
        print("   Baixe em: https://www.python.org/downloads/")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def install_requirements():
    """Instala as dependências do requirements.txt"""
    try:
        print("📦 Instalando dependências...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def check_camera():
    """Verifica se há uma câmera disponível"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print("✅ Câmera detectada e funcionando!")
                return True
        print("⚠️  Câmera não detectada ou não funcionando")
        return False
    except Exception as e:
        print(f"⚠️  Erro ao verificar câmera: {e}")
        return False

def check_model_files():
    """Verifica se os arquivos do modelo estão presentes"""
    model_files = [
        "modelos/modelo_aprimorado_20251001_115726.pkl",
        "modelos/modelo_info_aprimorado_20251001_115726.pkl"
    ]
    
    for file in model_files:
        if not os.path.exists(file):
            print(f"❌ Arquivo do modelo não encontrado: {file}")
            return False
    
    print("✅ Arquivos do modelo encontrados!")
    return True

def create_startup_script():
    """Cria script de inicialização para diferentes sistemas"""
    system = platform.system().lower()
    
    if system == "windows":
        script_content = """@echo off
echo Iniciando TraduLibras...
python app.py
pause
"""
        with open("start_tradulibras.bat", "w") as f:
            f.write(script_content)
        print("✅ Script de inicialização criado: start_tradulibras.bat")
    
    elif system in ["linux", "darwin"]:  # Linux ou Mac
        script_content = """#!/bin/bash
echo "Iniciando TraduLibras..."
python3 app.py
"""
        with open("start_tradulibras.sh", "w") as f:
            f.write(script_content)
        os.chmod("start_tradulibras.sh", 0o755)
        print("✅ Script de inicialização criado: start_tradulibras.sh")

def main():
    """Função principal de configuração"""
    print("🚀 Configuração Automática do TraduLibras")
    print("=" * 50)
    
    # Verificar Python
    if not check_python_version():
        return False
    
    # Instalar dependências
    if not install_requirements():
        return False
    
    # Verificar câmera
    check_camera()
    
    # Verificar modelos
    if not check_model_files():
        return False
    
    # Criar script de inicialização
    create_startup_script()
    
    print("\n🎉 Configuração concluída com sucesso!")
    print("\n📋 Próximos passos:")
    print("1. Execute: python app.py")
    print("2. Acesse: http://localhost:5000")
    print("3. Login: admin/admin123 ou user/user123")
    print("\n💡 Dica: Use o script de inicialização criado para facilitar!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

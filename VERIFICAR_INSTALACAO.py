#!/usr/bin/env python3
"""
Script de Verificação da Instalação TraduLibras
Verifica se todos os componentes necessários estão presentes
"""

import os
import sys
import subprocess

def print_header():
    print("=" * 60)
    print("VERIFICAÇÃO DA INSTALAÇÃO TRADULIBRAS")
    print("=" * 60)
    print()

def check_python():
    """Verificar versão do Python"""
    print("🐍 VERIFICANDO PYTHON...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Requer Python 3.8+)")
        return False

def check_files():
    """Verificar arquivos essenciais"""
    print("\n📁 VERIFICANDO ARQUIVOS ESSENCIAIS...")
    
    essential_files = [
        "app_funcional.py",
        "auth.py", 
        "requirements.txt",
        "users.json"
    ]
    
    essential_dirs = [
        "modelos",
        "templates", 
        "static",
        "static/css",
        "static/images"
    ]
    
    essential_templates = [
        "templates/login.html",
        "templates/admin_dashboard.html",
        "templates/camera_tradulibras.html"
    ]
    
    essential_models = [
        "modelos/modelo_inclusao_bc_20251003_144506.pkl",
        "modelos/scaler_inclusao_bc_20251003_144506.pkl", 
        "modelos/modelo_info_inclusao_bc_20251003_144506.pkl"
    ]
    
    all_ok = True
    
    # Verificar arquivos principais
    for file in essential_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (FALTANDO)")
            all_ok = False
    
    # Verificar diretórios
    for dirname in essential_dirs:
        if os.path.exists(dirname):
            print(f"✅ {dirname}/")
        else:
            print(f"❌ {dirname}/ (FALTANDO)")
            all_ok = False
    
    # Verificar templates
    for template in essential_templates:
        if os.path.exists(template):
            print(f"✅ {template}")
        else:
            print(f"❌ {template} (FALTANDO)")
            all_ok = False
    
    # Verificar modelos
    for model in essential_models:
        if os.path.exists(model):
            print(f"✅ {model}")
        else:
            print(f"❌ {model} (FALTANDO)")
            all_ok = False
    
    return all_ok

def check_dependencies():
    """Verificar dependências do Python"""
    print("\n📦 VERIFICANDO DEPENDÊNCIAS...")
    
    required_modules = [
        'flask', 'flask_login', 'cv2', 'mediapipe', 
        'numpy', 'sklearn', 'gtts', 'pandas'
    ]
    
    all_ok = True
    for module in required_modules:
        try:
            if module == 'cv2':
                import cv2
            elif module == 'flask_login':
                from flask_login import LoginManager
            elif module == 'sklearn':
                from sklearn.ensemble import RandomForestClassifier
            elif module == 'gtts':
                from gtts import gTTS
            else:
                __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} (FALTANDO)")
            all_ok = False
    
    return all_ok

def check_camera():
    """Verificar acesso à câmera"""
    print("\n📹 VERIFICANDO CÂMERA...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Testar leitura de um frame
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print("✅ Câmera detectada e funcionando")
                return True
            else:
                print("❌ Câmera detectada mas com erro de leitura")
                return False
        else:
            print("❌ Câmera não detectada")
            return False
    except Exception as e:
        print(f"❌ Erro ao verificar câmera: {e}")
        return False

def check_models():
    """Verificar modelos carregáveis"""
    print("\n🤖 VERIFICANDO MODELOS...")
    
    try:
        import pickle
        modelos_check = [
            'modelos/modelo_inclusao_bc_20251003_144506.pkl',
            'modelos/scaler_inclusao_bc_20251003_144506.pkl',
            'modelos/modelo_info_inclusao_bc_20251003_144506.pkl'
        ]
        
        all_ok = True
        for modelo in modelos_check:
            try:
                with open(modelo, 'rb') as f:
                    pickle.load(f)
                print(f"✅ {modelo} (válido)")
            except Exception as e:
                print(f"❌ {modelo} (erro: {e})")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"❌ Erro ao verificar modelos: {e}")
        return False

def check_gtts():
    """Verificar GTTS"""
    print("\n🔊 VERIFICANDO SISTEMA DE VOZ...")
    
    try:
        from gtts import gTTS
        print("✅ gTTS disponível")
        
        # Testar conexão com internet para gTTS
        import requests
        try:
            response = requests.get('https://translate.google.com', timeout=5)
            print("✅ Conexão com Google TTS disponível")
            return True
        except:
            print("⚠️ gTTS instalado mas sem conexão à internet")
            return False
            
    except Exception as e:
        print(f"❌ Erro no gTTS: {e}")
        return False

def main():
    """Função principal de verificação"""
    print_header()
    
    checks = [
        ("Python", check_python),
        ("Arquivos", check_files), 
        ("Dependências", check_dependencies),
        ("Câmera", check_camera),
        ("Modelos", check_models),
        ("Voz", check_gtts)
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Resumo final
    print("\n" + "=" * 60)
    print("RESUMO DA VERIFICAÇÃO")
    print("=" * 60)
    
    total_checks = len(checks)
    passed_checks = sum(results.values())
    
    for name, passed in results.items():
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{name:.<15} {status}")
    
    print(f"\nRESULTADO: {passed_checks}/{total_checks} verificações passaram")
    
    if passed_checks == total_checks:
        print("\n🎉 SISTEMA PRONTO PARA USO!")
        print("Execute: python app_funcional.py")
        print("Acesso: http://localhost:5000")
        print("Login: admin/admin123")
    else:
        print("\n⚠️ PROBLEMAS ENCONTRADOS!")
        print("Verifique os itens com ❌ acima")
        print("Execute o script de instalação ou contate o suporte")
    
    print("=" * 60)
    return passed_checks == total_checks

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

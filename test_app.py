#!/usr/bin/env python3
"""
Script de teste para o TraduLibras v2.0.0
Testa todas as funcionalidades principais da aplicação
"""

import requests
import time
import json
import sys

def test_app():
    """Testa todas as funcionalidades da aplicação"""
    base_url = "http://localhost:5000"
    
    print("🧪 Iniciando testes do TraduLibras v2.0.0...")
    print("=" * 50)
    
    # Teste 1: Status da aplicação
    print("1️⃣ Testando status da aplicação...")
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ Status: {status_data['status']}")
            print(f"📊 Modelo carregado: {status_data['model_loaded']}")
            print(f"🎤 Voz: {status_data['voice']}")
            print(f"📱 Versão: {status_data['version']}")
            print(f"🤖 Classes: {status_data['model_classes']}")
        else:
            print(f"❌ Erro no status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        return False
    
    # Teste 2: Página inicial
    print("\n2️⃣ Testando página inicial...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Página inicial carregada")
        else:
            print(f"❌ Erro na página inicial: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    # Teste 3: Página da câmera
    print("\n3️⃣ Testando página da câmera...")
    try:
        response = requests.get(f"{base_url}/camera", timeout=5)
        if response.status_code == 200:
            print("✅ Página da câmera carregada")
        else:
            print(f"❌ Erro na página da câmera: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    # Teste 4: Letra atual
    print("\n4️⃣ Testando rota de letra atual...")
    try:
        response = requests.get(f"{base_url}/letra_atual", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Letra atual: '{data.get('letra', 'Nenhuma')}'")
        else:
            print(f"❌ Erro na letra atual: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    # Teste 5: Limpeza de texto
    print("\n5️⃣ Testando limpeza de texto...")
    try:
        response = requests.get(f"{base_url}/clear_text", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Texto limpo: {data.get('message', 'OK')}")
        else:
            print(f"❌ Erro na limpeza: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    # Teste 6: Síntese de voz
    print("\n6️⃣ Testando síntese de voz...")
    try:
        test_text = "Teste de voz"
        response = requests.post(
            f"{base_url}/falar_texto",
            json={"texto": test_text},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print(f"✅ Síntese de voz funcionando: '{data.get('text_spoken')}'")
                print(f"🎤 Voz: {data.get('voice')}")
            else:
                print(f"⚠️ Aviso na síntese: {data.get('error')}")
        else:
            print(f"❌ Erro na síntese: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    # Teste 7: Stream de vídeo (verificar se existe)
    print("\n7️⃣ Testando stream de vídeo...")
    try:
        response = requests.get(f"{base_url}/video_feed", timeout=5)
        if response.status_code == 200:
            print("✅ Stream de vídeo disponível")
        else:
            print(f"❌ Erro no stream: {response.status_code}")
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Testes concluídos!")
    print("\n📋 Próximos passos:")
    print("1. Acesse http://localhost:5000 no navegador")
    print("2. Vá para a página da câmera")
    print("3. Permita acesso à webcam")
    print("4. Faça gestos das letras A, B, C, L, Y")
    print("5. Teste a síntese de voz com o botão 'Falar'")
    
    return True

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    print("🔍 Verificando dependências...")
    
    # Mapear nomes dos pacotes para imports
    package_imports = {
        'flask': 'flask',
        'opencv-python': 'cv2',
        'mediapipe': 'mediapipe',
        'scikit-learn': 'sklearn',
        'gtts': 'gtts',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NÃO INSTALADO")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Pacotes faltando: {', '.join(missing_packages)}")
        print("Execute: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ Todas as dependências estão instaladas!")
    return True

if __name__ == "__main__":
    print("🚀 TraduLibras v2.0.0 - Script de Teste")
    print("=" * 50)
    
    # Verificar dependências
    if not check_dependencies():
        sys.exit(1)
    
    print("\n⚠️ Certifique-se de que a aplicação está rodando em http://localhost:5000")
    input("Pressione Enter para continuar com os testes...")
    
    # Executar testes
    test_app()

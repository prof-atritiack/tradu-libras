#!/usr/bin/env python3
"""
Script de atualização para TraduLibras (Python Nativo)
Autor: TraduLibras Team
Versão: 2.0.0
"""

import os
import sys
import subprocess
import shutil
import datetime
from pathlib import Path

# Cores para output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_message(message):
    print(f"{Colors.GREEN}[TraduLibras]{Colors.NC} {message}")

def print_warning(message):
    print(f"{Colors.YELLOW}[AVISO]{Colors.NC} {message}")

def print_error(message):
    print(f"{Colors.RED}[ERRO]{Colors.NC} {message}")

def print_info(message):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

def check_git_repo():
    """Verifica se é um repositório Git"""
    try:
        subprocess.run(['git', 'status'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def backup_models():
    """Faz backup dos modelos treinados"""
    print_info("Fazendo backup dos modelos...")
    
    backup_dir = Path("backup")
    backup_dir.mkdir(exist_ok=True)
    
    models_dir = Path("modelos")
    if models_dir.exists():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_file in models_dir.glob("*.pkl"):
            backup_file = backup_dir / f"{model_file.stem}_backup_{timestamp}.pkl"
            shutil.copy2(model_file, backup_file)
            print_info(f"Backup salvo: {backup_file}")
    
    # Backup do arquivo de dados
    data_file = Path("gestos_libras.csv")
    if data_file.exists():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"gestos_libras_backup_{timestamp}.csv"
        shutil.copy2(data_file, backup_file)
        print_info(f"Backup salvo: {backup_file}")

def update_dependencies():
    """Atualiza as dependências Python"""
    print_info("Atualizando dependências...")
    
    try:
        # Atualizar pip
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True)
        
        # Instalar/atualizar dependências
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        
        print_message("Dependências atualizadas com sucesso! ✅")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Erro ao atualizar dependências: {e}")
        return False

def check_for_updates():
    """Verifica se há atualizações disponíveis"""
    try:
        # Buscar atualizações
        subprocess.run(['git', 'fetch', 'origin'], check=True, capture_output=True)
        
        # Verificar se há atualizações
        result = subprocess.run(['git', 'status', '-uno'], 
                               capture_output=True, text=True, check=True)
        
        return "behind" in result.stdout
    except subprocess.CalledProcessError:
        return False

def apply_updates():
    """Aplica as atualizações do repositório"""
    try:
        # Fazer pull das atualizações
        subprocess.run(['git', 'pull', 'origin', 'main'], check=True, capture_output=True)
        print_message("Atualizações aplicadas com sucesso! ✅")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Erro ao aplicar atualizações: {e}")
        return False

def retrain_model():
    """Pergunta se deve retreinar o modelo"""
    print_warning("Deseja retreinar o modelo com os novos dados? (Y/n)")
    response = input().strip().lower()
    
    if response not in ['n', 'no']:
        print_info("Retreinando modelo...")
        try:
            subprocess.run([sys.executable, 'treinar_letras_simples.py'], check=True)
            print_message("Modelo retreinado com sucesso! ✅")
        except subprocess.CalledProcessError as e:
            print_error(f"Erro ao retreinar modelo: {e}")
            return False
    return True

def main():
    """Função principal"""
    print(f"{Colors.BLUE}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                🤟 TraduLibras Update 🤟                      ║")
    print("║              Sistema de Atualização Automática               ║")
    print("║                        Versão 2.0.0                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.NC}")
    
    # Verificar se é um repositório Git
    if not check_git_repo():
        print_warning("Este não é um repositório Git.")
        print_info("Para atualizações automáticas, clone o projeto do GitHub:")
        print_info("git clone https://github.com/prof-atritiack/libras-js.git")
        print()
        print_info("Ou baixe manualmente as atualizações do GitHub.")
        return 1
    
    # Fazer backup dos modelos
    backup_models()
    
    # Verificar se há atualizações
    print_info("Verificando atualizações...")
    if not check_for_updates():
        print_info("Projeto já está atualizado! ✅")
        print_info("Atualizando dependências...")
        update_dependencies()
        return 0
    
    # Aplicar atualizações
    print_info("Atualizações encontradas! Aplicando...")
    if not apply_updates():
        return 1
    
    # Atualizar dependências
    if not update_dependencies():
        return 1
    
    # Perguntar sobre retreinar modelo
    if not retrain_model():
        return 1
    
    print()
    print_message("Atualização concluída com sucesso! 🎉")
    print_info("Execute 'python app.py' para iniciar a versão atualizada!")
    print()
    print_info("📱 Acesse: http://localhost:5000")
    print_info("🌐 Para rede local, use o botão 'Info Rede' na interface")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_error("\nAtualização cancelada pelo usuário.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Erro inesperado: {e}")
        sys.exit(1)

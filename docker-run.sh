#!/bin/bash

# Script para executar TraduLibras com Docker
# Autor: TraduLibras Team
# Versão: 2.0.0

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para imprimir mensagens coloridas
print_message() {
    echo -e "${GREEN}[TraduLibras]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[AVISO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERRO]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    🤟 TraduLibras Docker 🤟                  ║"
echo "║              Sistema de Reconhecimento de LIBRAS             ║"
echo "║                        Versão 2.0.0                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Verificar se Docker está instalado
if ! command -v docker &> /dev/null; then
    print_error "Docker não está instalado!"
    print_info "Instale o Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Verificar se Docker Compose está instalado
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose não está instalado!"
    print_info "Instale o Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Verificar se Docker está rodando
if ! docker info &> /dev/null; then
    print_error "Docker não está rodando!"
    print_info "Inicie o Docker Desktop ou Docker daemon"
    exit 1
fi

print_message "Verificações iniciais concluídas ✅"

# Função para obter IP local
get_local_ip() {
    if command -v ip &> /dev/null; then
        ip route get 1.1.1.1 | awk '{print $7; exit}'
    elif command -v hostname &> /dev/null; then
        hostname -I | awk '{print $1}'
    else
        echo "localhost"
    fi
}

# Função para mostrar informações de acesso
show_access_info() {
    local_ip=$(get_local_ip)
    
    echo -e "\n${GREEN}🚀 TraduLibras está rodando!${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}📱 ACESSO LOCAL:${NC}"
    echo -e "   http://localhost:5000"
    echo -e "   http://127.0.0.1:5000"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}🌐 ACESSO NA REDE LOCAL:${NC}"
    echo -e "   http://${local_ip}:5000"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}📱 Para acessar de outros dispositivos:${NC}"
    echo -e "   1. Certifique-se que estão na mesma rede Wi-Fi"
    echo -e "   2. Use o endereço: http://${local_ip}:5000"
    echo -e "   3. Funciona em celular, tablet e outros computadores"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}🛠️  Comandos úteis:${NC}"
    echo -e "   • Parar: docker-compose down"
    echo -e "   • Logs: docker-compose logs -f"
    echo -e "   • Status: docker-compose ps"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

# Função para treinar modelo
train_model() {
    print_info "Iniciando treinamento do modelo..."
    
    # Verificar se já existe um modelo
    if [ -f "modelos/modelo_libras.pkl" ]; then
        print_warning "Modelo já existe. Deseja retreinar? (y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            print_info "Usando modelo existente."
            return 0
        fi
    fi
    
    print_info "Executando treinamento do modelo..."
    docker-compose run --rm tradulibras python treinar_letras_simples.py
    
    if [ $? -eq 0 ]; then
        print_message "Modelo treinado com sucesso! ✅"
    else
        print_error "Erro no treinamento do modelo!"
        exit 1
    fi
}

# Função para mostrar ajuda
show_help() {
    echo -e "${YELLOW}Uso: $0 [OPÇÃO]${NC}"
    echo ""
    echo "Opções:"
    echo "  start, s     Iniciar o TraduLibras (padrão)"
    echo "  stop         Parar o TraduLibras"
    echo "  restart      Reiniciar o TraduLibras"
    echo "  build        Construir a imagem Docker"
    echo "  train        Treinar o modelo de reconhecimento"
    echo "  logs         Mostrar logs do container"
    echo "  status       Mostrar status do container"
    echo "  update       Atualizar projeto do GitHub"
    echo "  clean        Limpar containers e imagens"
    echo "  help, h      Mostrar esta ajuda"
    echo ""
    echo "Exemplos:"
    echo "  $0 start     # Iniciar o TraduLibras"
    echo "  $0 train     # Treinar o modelo"
    echo "  $0 logs      # Ver logs em tempo real"
}

# Função para limpar containers
clean_docker() {
    print_warning "Isso irá remover todos os containers e imagens do TraduLibras."
    print_warning "Deseja continuar? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Limpando containers e imagens..."
        docker-compose down --rmi all --volumes --remove-orphans
        docker system prune -f
        print_message "Limpeza concluída! ✅"
    else
        print_info "Operação cancelada."
    fi
}

# Função para atualizar o projeto
update_project() {
    print_message "Atualizando projeto TraduLibras..."
    echo
    
    # Verificar se é um repositório Git
    if ! git status &> /dev/null; then
        print_warning "Este não é um repositório Git."
        print_info "Para atualizações automáticas, clone o projeto do GitHub:"
        print_info "git clone https://github.com/prof-atritiack/libras-js.git"
        echo
        print_info "Ou baixe manualmente as atualizações do GitHub."
        return 1
    fi
    
    # Fazer backup dos modelos
    print_info "Fazendo backup dos modelos..."
    if [ -f "modelos/modelo_libras.pkl" ]; then
        mkdir -p backup
        cp "modelos/modelo_libras.pkl" "backup/modelo_libras_backup_$(date +%Y%m%d_%H%M%S).pkl"
        print_info "Backup do modelo salvo em backup/"
    fi
    
    # Parar containers se estiverem rodando
    print_info "Parando containers..."
    docker-compose down &> /dev/null
    
    # Atualizar código do repositório
    print_info "Baixando atualizações do GitHub..."
    if ! git fetch origin; then
        print_error "Erro ao buscar atualizações!"
        return 1
    fi
    
    # Verificar se há atualizações
    if ! git status -uno | grep -q "behind"; then
        print_info "Projeto já está atualizado! ✅"
        print_info "Reiniciando containers..."
        docker-compose up -d
        show_access_info
        return 0
    fi
    
    print_info "Atualizações encontradas! Aplicando..."
    if ! git pull origin main; then
        print_error "Erro ao aplicar atualizações!"
        print_info "Verifique se há conflitos ou se o repositório está limpo."
        return 1
    fi
    
    print_message "Atualizações aplicadas com sucesso! ✅"
    
    # Reconstruir imagem Docker
    print_info "Reconstruindo imagem Docker..."
    if ! docker-compose build --no-cache; then
        print_error "Erro ao reconstruir a imagem!"
        return 1
    fi
    
    # Iniciar containers
    print_info "Iniciando containers atualizados..."
    if ! docker-compose up -d; then
        print_error "Erro ao iniciar containers!"
        return 1
    fi
    
    print_info "Aguardando containers estarem prontos..."
    sleep 15
    
    # Verificar se está rodando
    if docker-compose ps | grep -q "Up"; then
        echo
        print_message "Atualização concluída com sucesso! 🎉"
        print_info "Versão atualizada do TraduLibras está rodando!"
        echo
        show_access_info
    else
        print_error "Erro ao iniciar containers atualizados!"
        print_info "Verifique os logs: docker-compose logs"
        return 1
    fi
}

# Processar argumentos
case "${1:-start}" in
    "start"|"s"|"")
        print_message "Iniciando TraduLibras com Docker..."
        
        # Verificar se precisa treinar modelo
        if [ ! -f "modelos/modelo_libras.pkl" ]; then
            print_warning "Modelo não encontrado. Deseja treinar agora? (Y/n)"
            read -r response
            if [[ ! "$response" =~ ^[Nn]$ ]]; then
                train_model
            fi
        fi
        
        # Construir e iniciar
        print_info "Construindo imagem Docker..."
        docker-compose build
        
        print_info "Iniciando container..."
        docker-compose up -d
        
        # Aguardar container estar pronto
        print_info "Aguardando container estar pronto..."
        sleep 10
        
        # Verificar se está rodando
        if docker-compose ps | grep -q "Up"; then
            show_access_info
        else
            print_error "Erro ao iniciar o container!"
            print_info "Verifique os logs: docker-compose logs"
            exit 1
        fi
        ;;
        
    "update")
        update_project
        ;;
        
    "stop")
        print_message "Parando TraduLibras..."
        docker-compose down
        print_message "TraduLibras parado! ✅"
        ;;
        
    "restart")
        print_message "Reiniciando TraduLibras..."
        docker-compose restart
        show_access_info
        ;;
        
    "build")
        print_message "Construindo imagem Docker..."
        docker-compose build --no-cache
        print_message "Imagem construída! ✅"
        ;;
        
    "train")
        train_model
        ;;
        
    "logs")
        print_info "Mostrando logs do TraduLibras..."
        docker-compose logs -f
        ;;
        
    "status")
        print_info "Status do TraduLibras:"
        docker-compose ps
        ;;
        
    "clean")
        clean_docker
        ;;
        
    "help"|"h")
        show_help
        ;;
        
    *)
        print_error "Opção inválida: $1"
        show_help
        exit 1
        ;;
esac

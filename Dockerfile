# Dockerfile para TraduLibras
# Baseado em Python 3.10 slim para menor tamanho
FROM python:3.10-slim

# Metadados do container
LABEL maintainer="TraduLibras Team"
LABEL description="Sistema de reconhecimento de LIBRAS com IA"
LABEL version="2.0.0"

# Definir variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências do sistema necessárias para OpenCV e MediaPipe
RUN apt-get update && apt-get install -y \
    # Dependências básicas
    build-essential \
    cmake \
    pkg-config \
    # Dependências para OpenCV
    libopencv-dev \
    python3-opencv \
    # Dependências para MediaPipe
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Dependências para áudio
    libasound2-dev \
    portaudio19-dev \
    # Dependências para rede
    net-tools \
    iputils-ping \
    # Utilitários
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependências primeiro (para cache do Docker)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY . .

# Criar diretórios necessários
RUN mkdir -p modelos static/css static/images templates

# Definir permissões
RUN chmod +x *.py

# Expor porta
EXPOSE 5000

# Criar usuário não-root para segurança
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Comando de inicialização
CMD ["python", "app.py"]

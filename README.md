# 🤟 TraduLibras - Sistema de Reconhecimento de LIBRAS com IA

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/mediapipe-informational?logo=google" />
  <img src="https://img.shields.io/badge/opencv-4.x-green?logo=opencv" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn" />
  <img src="https://img.shields.io/badge/cursor-ai-purple?logo=cursor" />
  <img src="https://img.shields.io/badge/status-produção-brightgreen" />
</p>

Sistema completo de reconhecimento de Língua Brasileira de Sinais (LIBRAS) usando **visão computacional**, **machine learning** e **síntese de voz**. Este projeto é ideal para desenvolvedores iniciantes que querem aprender sobre IA, acessibilidade e inclusão digital.

## 🎯 O que este projeto faz?

- **Reconhece gestos de LIBRAS** em tempo real usando sua webcam
- **Converte gestos em texto** usando inteligência artificial
- **Fala o texto reconhecido** usando síntese de voz
- **Corrige automaticamente** erros de reconhecimento
- **Interface web moderna** e responsiva
- **Animações visuais** para feedback do usuário

## 🚀 Guia Completo para Iniciantes

### 📋 Pré-requisitos (O que você precisa)

#### Hardware:
- 💻 **Computador** com Windows, Mac ou Linux
- 📹 **Webcam** funcionando (qualquer resolução)
- 🌐 **Internet** para instalação inicial
- 💾 **2GB de espaço livre** no disco

#### Software (Escolha uma opção):

##### 🐳 **Opção 1: Docker (Recomendado para Iniciantes)**
- 🐳 **Docker Desktop** ([Download aqui](https://www.docker.com/products/docker-desktop/))
- 🔧 **Git** ([Download aqui](https://git-scm.com/downloads)) - Para clonar o projeto

##### 🐍 **Opção 2: Python Nativo**
- 🐍 **Python 3.10 ou superior** ([Download aqui](https://www.python.org/downloads/))
- 📝 **Cursor AI** ([Download aqui](https://cursor.sh/)) - Editor de código com IA
- 🔧 **Git** ([Download aqui](https://git-scm.com/downloads)) - Para clonar o projeto

### 🛠️ Instalação Passo a Passo

## 🐳 **MÉTODO 1: Docker (Mais Fácil para Iniciantes)**

### Passo 1: Instalar Docker Desktop
1. Acesse [docker.com](https://www.docker.com/products/docker-desktop/)
2. Baixe o Docker Desktop para seu sistema operacional
3. Instale e inicie o Docker Desktop
4. Verifique a instalação abrindo o terminal e digitando:
   ```bash
   docker --version
   docker-compose --version
   ```

### Passo 2: Clonar o Projeto
```bash
git clone https://github.com/prof-atritiack/libras-js.git
cd libras-js
```

### Passo 3: Executar com Docker
```bash
# Windows
docker-run.bat start

# Linux/Mac
./docker-run.sh start
```

**🎉 Pronto! O TraduLibras estará rodando em http://localhost:5000**

### 🎯 **Vantagens do Docker:**
- ✅ **Instalação em 1 comando** - sem configurar Python, dependências, etc.
- ✅ **Funciona em qualquer sistema** - Windows, Mac, Linux
- ✅ **Ambiente isolado** - não interfere com outros projetos
- ✅ **Fácil de remover** - delete o container e pronto
- ✅ **Mesmo ambiente** - funciona igual para todos
- ✅ **Atualizações automáticas** - sempre usa as versões corretas

### Comandos Docker Úteis:
```bash
# Iniciar
docker-run.bat start          # Windows
./docker-run.sh start         # Linux/Mac

# Atualizar projeto
docker-run.bat update         # Windows
./docker-run.sh update        # Linux/Mac

# Treinar modelo
docker-run.bat train          # Windows
./docker-run.sh train         # Linux/Mac

# Ver logs
docker-run.bat logs           # Windows
./docker-run.sh logs          # Linux/Mac

# Parar
docker-run.bat stop           # Windows
./docker-run.sh stop          # Linux/Mac

# Limpar tudo
docker-run.bat clean          # Windows
./docker-run.sh clean         # Linux/Mac
```

### 🛠️ **Para Desenvolvedores:**
```bash
# Modo desenvolvimento (com hot reload)
docker-compose -f docker-compose.dev.yml up

# Construir imagem personalizada
docker build -t tradulibras-custom .

# Executar comando personalizado
docker-compose run --rm tradulibras python seu_script.py

# Acessar shell do container
docker-compose exec tradulibras bash
```

---

## 🔄 **Como Atualizar o Projeto**

### 🐳 **Com Docker (Recomendado):**
```bash
# Atualizar automaticamente
docker-run.bat update         # Windows
./docker-run.sh update        # Linux/Mac
```

**O que o comando `update` faz:**
- ✅ **Backup automático** dos modelos treinados
- ✅ **Baixa atualizações** do GitHub
- ✅ **Reconstrói a imagem** Docker
- ✅ **Reinicia containers** com nova versão
- ✅ **Mantém seus dados** (modelos, configurações)

### 🐍 **Com Python Nativo:**
```bash
# Atualizar automaticamente
update-project.bat            # Windows
python update-project.py      # Linux/Mac
```

**O que o script de atualização faz:**
- ✅ **Backup automático** dos modelos e dados
- ✅ **Atualiza código** do GitHub
- ✅ **Atualiza dependências** Python
- ✅ **Pergunta se quer retreinar** o modelo
- ✅ **Mantém ambiente virtual** intacto

### 📋 **Atualização Manual (se necessário):**
```bash
# 1. Fazer backup
cp -r modelos backup/modelos_$(date +%Y%m%d)
cp gestos_libras.csv backup/gestos_libras_$(date +%Y%m%d).csv

# 2. Atualizar código
git pull origin main

# 3. Atualizar dependências
pip install -r requirements.txt

# 4. Retreinar modelo (opcional)
python treinar_letras_simples.py
```

---

## 🐍 **MÉTODO 2: Python Nativo**

### Passo 1: Instalar Python
1. Acesse [python.org](https://www.python.org/downloads/)
2. Baixe a versão mais recente (3.10+)
3. **IMPORTANTE**: Durante a instalação, marque "Add Python to PATH"
4. Verifique a instalação abrindo o terminal e digitando:
   ```bash
   python --version
   ```

#### Passo 2: Instalar Cursor AI
1. Acesse [cursor.sh](https://cursor.sh/)
2. Baixe e instale o Cursor AI
3. Crie uma conta gratuita
4. O Cursor AI é um editor de código com IA integrada que vai te ajudar muito!

#### Passo 3: Clonar o Projeto
1. Abra o terminal (PowerShell no Windows, Terminal no Mac/Linux)
2. Navegue até a pasta onde quer salvar o projeto:
   ```bash
   cd Desktop  # ou qualquer pasta de sua escolha
   ```
3. Clone o projeto:
   ```bash
   git clone https://github.com/prof-atritiack/libras-js.git
   cd libras-js
   ```

#### Passo 4: Configurar Ambiente Virtual
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# No Windows:
venv\Scripts\activate

# No Mac/Linux:
source venv/bin/activate
```

#### Passo 5: Instalar Dependências
```bash
pip install -r requirements.txt
```

### 🎓 Usando o Cursor AI para Desenvolvimento

#### O que é o Cursor AI?
O Cursor AI é um editor de código que tem **inteligência artificial integrada**. Ele pode:
- ✨ **Explicar código** que você não entende
- 🔧 **Corrigir erros** automaticamente
- 📝 **Escrever código** baseado em suas descrições
- 🐛 **Debugar problemas** e sugerir soluções
- 📚 **Ensinar conceitos** de programação

#### Como usar o Cursor AI neste projeto:

1. **Abra o projeto no Cursor AI:**
   ```bash
   # No terminal, dentro da pasta do projeto:
   cursor .
   ```

2. **Comandos úteis do Cursor AI:**
   - `Ctrl+K` (Windows) ou `Cmd+K` (Mac): Abre o chat com IA
   - `Ctrl+L` (Windows) ou `Cmd+L` (Mac): Abre chat lateral
   - `Ctrl+I` (Windows) ou `Cmd+I` (Mac): Edição inline com IA

3. **Exemplos de perguntas para fazer ao Cursor AI:**
   ```
   "Explique como funciona o arquivo app.py"
   "Por que o MediaPipe é usado neste projeto?"
   "Como posso melhorar a precisão do reconhecimento?"
   "Me ajude a entender o código de machine learning"
   ```

4. **Dicas para usar o Cursor AI:**
   - Seja específico nas suas perguntas
   - Peça para explicar código linha por linha
   - Use para corrigir erros que aparecerem
   - Peça sugestões de melhorias

### 🎯 Treinamento do Modelo (Coleta de Dados)

#### Passo 1: Executar Coleta de Dados
```bash
# Certifique-se que o ambiente virtual está ativo
python treinar_letras_simples.py
```

#### Passo 2: Processo de Coleta
1. **A câmera será ativada automaticamente**
2. **Para cada letra (A, B, C, L, Y):**
   - Você verá o nome da letra na tela
   - Posicione sua mão no centro da câmera
   - Faça o gesto da letra correspondente
   - Pressione **ESPAÇO** para capturar uma amostra
   - Pressione **ESC** para pular uma letra

3. **Dicas importantes:**
   - 📸 **Colete 30-50 amostras por letra**
   - 🌞 **Use boa iluminação**
   - 📏 **Mantenha a mão a ~50cm da câmera**
   - 🔄 **Varie posições e ângulos**
   - ⏸️ **Evite movimentos durante a captura**

#### Passo 3: Treinamento Automático
Após coletar os dados, o modelo será treinado automaticamente:
- ✅ Divisão dos dados (80% treino, 20% teste)
- ✅ Treinamento do modelo Random Forest
- ✅ Avaliação da acurácia
- ✅ Salvamento do modelo em `modelos/modelo_libras.pkl`

### 🚀 Executando o Projeto

#### Passo 1: Ativar Ambiente Virtual
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

#### Passo 2: Executar Aplicação
```bash
python app.py
```

#### Passo 3: Acessar no Navegador
```
http://localhost:5000
```

### 🎮 Como Usar o Sistema

1. **Acesse a página da câmera**
2. **Permita acesso à webcam** quando solicitado
3. **Posicione sua mão** no centro da câmera
4. **Faça gestos de LIBRAS** para as letras A, B, C, L, Y
5. **Veja o texto sendo formado** em tempo real
6. **Use o botão "Falar"** para ouvir o texto
7. **Use "Limpar"** para recomeçar

### 🔧 Solução de Problemas Comuns

#### ❌ Problema: "python não é reconhecido"
**Solução:**
1. Reinstale o Python marcando "Add to PATH"
2. Ou use `python3` em vez de `python`

#### ❌ Problema: Webcam não funciona
**Solução:**
1. Verifique se a webcam está conectada
2. Feche outros programas que usam a câmera
3. Recarregue a página
4. Reinicie a aplicação

#### ❌ Problema: Erro ao instalar dependências
**Solução:**
```bash
# Atualize o pip
python -m pip install --upgrade pip

# Instale uma dependência por vez
pip install flask
pip install opencv-python
pip install mediapipe
pip install scikit-learn
pip install gtts
```

#### ❌ Problema: Reconhecimento impreciso
**Solução:**
1. **Retreine o modelo** com mais amostras
2. **Melhore a iluminação**
3. **Mantenha gestos consistentes**
4. **Evite movimentos bruscos**

### 🎨 Funcionalidades Avançadas

#### ✨ Efeitos Visuais
- **Animações de detecção** com mudança de cores
- **Feedback visual** quando letras são reconhecidas
- **Interface responsiva** que funciona em mobile
- **Efeitos de hover** e transições suaves

#### 🧠 Correção Automática de Texto
- **Dicionário inteligente** com palavras comuns
- **Correção de erros** usando distância de Levenshtein
- **Contador de correções** aplicadas
- **Tooltips** mostrando texto original

#### 🔊 Síntese de Voz
- **Reprodução direta no navegador** (sem abrir aplicativos externos)
- **Voz em português brasileiro** usando gTTS
- **Controle de áudio** integrado

### 📁 Estrutura do Projeto

```
tradu-libras/
├── 📄 app.py                          # Aplicação Flask principal
├── 🎯 treinar_letras_simples.py       # Script de coleta e treinamento
├── 📊 gestos_libras.csv               # Dataset de gestos coletados
├── 🤖 modelo_libras.pkl               # Modelo treinado (raiz)
├── 📁 modelos/                        # Diretório de modelos
│   ├── 🤖 modelo_libras.pkl           # Modelo treinado (atual)
│   └── ℹ️ modelo_info.pkl             # Informações do modelo
├── 📋 requirements.txt                # Dependências Python
├── 📁 templates/                      # Templates HTML
│   ├── 🏠 index.html                  # Página inicial
│   ├── 📹 camera_tradulibras.html     # Interface principal
│   ├── 📚 tutorial.html               # Tutorial do sistema
│   └── ⚙️ configuracoes.html          # Configurações
├── 📁 static/                         # Arquivos estáticos
│   ├── 🎨 css/                        # Estilos CSS
│   └── 🖼️ images/                     # Imagens e ícones
└── 📖 README.md                       # Esta documentação
```

### 🛠️ Tecnologias Utilizadas

| Tecnologia | Função | Por que usar? |
|------------|--------|---------------|
| **Flask** | Framework web | Simples e poderoso para iniciantes |
| **OpenCV** | Processamento de imagem | Padrão da indústria para visão computacional |
| **MediaPipe** | Detecção de mãos | Desenvolvido pelo Google, muito preciso |
| **Scikit-learn** | Machine Learning | Biblioteca mais popular para ML em Python |
| **gTTS** | Síntese de voz | Gratuito e de alta qualidade |
| **HTML/CSS/JS** | Interface | Tecnologias web padrão |

### 🎓 Conceitos de IA que você vai aprender

#### 1. **Visão Computacional**
- Como câmeras capturam imagens
- Processamento de imagens em tempo real
- Detecção de objetos (mãos)

#### 2. **Machine Learning**
- O que são features (características)
- Como treinar modelos
- Validação e teste de modelos
- Algoritmo Random Forest

#### 3. **Processamento de Dados**
- Normalização de dados
- Divisão treino/teste
- Métricas de avaliação (acurácia)

#### 4. **Desenvolvimento Web**
- APIs REST
- Comunicação frontend/backend
- Streaming de vídeo
- Síntese de voz

### 🚀 Próximos Passos (Para Aprender Mais)

#### Nível Iniciante:
1. **Modifique as letras reconhecidas** - adicione novas letras
2. **Mude as cores da interface** - personalize o visual
3. **Adicione novos efeitos visuais** - crie animações

#### Nível Intermediário:
1. **Implemente reconhecimento de palavras completas**
2. **Adicione mais gestos** (números, sinais básicos)
3. **Melhore a precisão** com mais dados de treinamento

#### Nível Avançado:
1. **Use redes neurais** (TensorFlow/PyTorch)
2. **Implemente reconhecimento de frases**
3. **Adicione tradução para outras línguas de sinais**

### 🤝 Como Contribuir

#### Para Iniciantes:
1. **Teste o projeto** e reporte bugs
2. **Melhore a documentação** com suas descobertas
3. **Adicione exemplos** de uso

#### Para Desenvolvedores:
1. **Fork o projeto**
2. **Crie uma branch** para sua feature
3. **Faça commit** das mudanças
4. **Abra um Pull Request**

### 📞 Suporte e Comunidade

#### Se você tiver dúvidas:
1. **Use o Cursor AI** - ele pode explicar qualquer parte do código
2. **Leia a documentação** - este README tem tudo que você precisa
3. **Procure por issues** no GitHub
4. **Crie uma nova issue** se não encontrar a solução

#### Recursos de Aprendizado:
- 📚 [Documentação do Python](https://docs.python.org/)
- 🎥 [Tutoriais de OpenCV](https://opencv.org/tutorials/)
- 🤖 [Guia de Scikit-learn](https://scikit-learn.org/stable/user_guide.html)
- 🌐 [Documentação do Flask](https://flask.palletsprojects.com/)

### 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 👥 Equipe

Desenvolvido com ❤️ pela equipe TraduLibras para promover **acessibilidade** e **inclusão digital**.

---

## 🎯 Resumo Rápido para Começar

### 🐳 **Com Docker (Recomendado):**
```bash
# 1. Clone o projeto
git clone https://github.com/prof-atritiack/libras-js.git
cd libras-js

# 2. Execute com Docker
docker-run.bat start          # Windows
./docker-run.sh start         # Linux/Mac

# 3. Acesse no navegador
# http://localhost:5000

# 4. Para atualizar (quando houver novas versões)
docker-run.bat update         # Windows
./docker-run.sh update        # Linux/Mac
```

### 🐍 **Com Python Nativo:**
```bash
# 1. Clone o projeto
git clone https://github.com/prof-atritiack/libras-js.git
cd libras-js

# 2. Crie ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Instale dependências
pip install -r requirements.txt

# 4. Treine o modelo
python treinar_letras_simples.py

# 5. Execute a aplicação
python app.py

# 6. Acesse no navegador
# http://localhost:5000

# 7. Para atualizar (quando houver novas versões)
update-project.bat            # Windows
python update-project.py      # Linux/Mac
```

**🎉 Pronto! Você tem um sistema de reconhecimento de LIBRAS funcionando!**

---

*Este projeto foi criado para ser acessível a desenvolvedores de todos os níveis. Se você é iniciante, não se preocupe - use o Cursor AI para tirar dúvidas e aprender no processo!*
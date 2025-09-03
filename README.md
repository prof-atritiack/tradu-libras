# 🤟 TraduLibras - Reconhecimento de Gestos em Libras

> **Sistema inteligente de reconhecimento de gestos em Libras usando IA e visão computacional**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0.2-green.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-red.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-orange.svg)](https://mediapipe.dev/)

## 🎯 **O que é o TraduLibras?**

O **TraduLibras** é um sistema avançado que utiliza **Inteligência Artificial** e **visão computacional** para reconhecer gestos em **Libras (Língua Brasileira de Sinais)** em tempo real. O sistema converte os gestos capturados pela webcam em texto e áudio, facilitando a comunicação entre pessoas surdas e ouvintes.

### ✨ **Principais Funcionalidades:**

- 🎥 **Reconhecimento em tempo real** de gestos em Libras
- 🔊 **Síntese de voz** automática do texto formado
- 🧠 **Correção automática** de texto com IA
- 🌐 **Interface web responsiva** e intuitiva
- 📱 **Acesso via rede local** para múltiplos dispositivos
- 🔐 **Sistema de autenticação** com diferentes níveis de acesso
- 📊 **Painel administrativo** para gerenciamento do sistema

## 🚀 **Demonstração Rápida**

```bash
# 1. Clone o projeto
git clone https://github.com/prof-atritiack/libras-js.git
cd libras-js

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute a aplicação
python app.py

# 4. Acesse no navegador
# http://localhost:5000
```

## 📋 **Requisitos do Sistema**

### 💻 **Hardware:**
- 💻 **Computador** com Windows, Mac ou Linux
- 📹 **Webcam** funcionando (qualquer resolução)
- 🌐 **Internet** para instalação inicial
- 💾 **2GB de espaço livre** no disco

### 🛠️ **Software Necessário:**
- 🐍 **Python 3.10 ou superior** ([Download aqui](https://www.python.org/downloads/))
- 📝 **Cursor AI** ([Download aqui](https://cursor.sh/)) - Editor de código com IA
- 🔧 **Git** ([Download aqui](https://git-scm.com/downloads)) - Para clonar o projeto

## 🛠️ **Instalação Passo a Passo**

### **Passo 1: Instalar Python**
1. Acesse [python.org](https://www.python.org/downloads/)
2. Baixe a versão 3.10 ou superior para seu sistema
3. **IMPORTANTE**: Durante a instalação, marque "Add Python to PATH"
4. Verifique a instalação abrindo o terminal e digitando:
   ```bash
   python --version
   pip --version
   ```

### **Passo 2: Clonar o Projeto**
```bash
git clone https://github.com/prof-atritiack/libras-js.git
cd libras-js
```

### **Passo 3: Criar Ambiente Virtual (Recomendado)**
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### **Passo 4: Instalar Dependências**
```bash
pip install -r requirements.txt
```

### **Passo 5: Executar a Aplicação**
```bash
python app.py
```

**🎉 Pronto! O TraduLibras estará rodando em http://localhost:5000**

## 🔐 **Sistema de Autenticação**

O TraduLibras possui um sistema completo de autenticação com diferentes níveis de acesso:

### 👥 **Tipos de Usuário:**

#### **🔑 Administrador (Admin):**
- ✅ **Acesso completo** ao sistema
- ✅ **Painel administrativo** com funcionalidades avançadas
- ✅ **Gerenciar usuários** (criar, editar, remover)
- ✅ **Atualizar sistema** diretamente pela interface
- ✅ **Criar backups** e gerenciar logs
- ✅ **Acesso a todas as funcionalidades** do TraduLibras

#### **👤 Usuário Padrão:**
- ✅ **Acesso ao reconhecimento** de gestos
- ✅ **Usar todas as funcionalidades** principais
- ✅ **Síntese de voz** e correção de texto
- ✅ **Acesso via rede local**

### 🔑 **Credenciais Padrão:**
```
👑 Administrador:
   Usuário: admin
   Senha: admin123

👤 Usuário:
   Usuário: user
   Senha: user123
```

## 🎮 **Como Usar**

### **1. Fazer Login**
- Acesse http://localhost:5000
- Use as credenciais fornecidas acima
- Escolha entre usuário comum ou administrador

### **2. Reconhecer Gestos**
- Posicione sua mão na frente da webcam
- Faça os gestos das letras: **A, B, C, L, Y**
- O sistema reconhecerá e formará palavras automaticamente
- Use o botão "Reproduzir Texto" para ouvir o resultado

### **3. Funcionalidades Disponíveis**
- 🎤 **Falar Letra**: Reproduz apenas a letra detectada
- 🔊 **Reproduzir Texto**: Converte todo o texto em áudio
- ✏️ **Simular Texto**: Testa o sistema com exemplos
- 🧪 **Testar Correção**: Verifica a correção automática
- 🗑️ **Limpar Texto**: Reseta todo o texto formado
- 🌐 **Info Rede**: Mostra URLs para acesso via rede local

## 🌐 **Acesso via Rede Local**

O TraduLibras pode ser acessado de outros dispositivos na mesma rede:

### **Para acessar de outros dispositivos:**
1. **Celular/Tablet**: Use o endereço IP mostrado na tela
2. **Outro computador**: Acesse o mesmo endereço IP
3. **Requisitos**: Todos os dispositivos devem estar na mesma rede Wi-Fi

### **Exemplo de URLs:**
```
💻 Local: http://localhost:5000
📱 Rede: http://192.168.1.100:5000
```

## 🧠 **Tecnologias Utilizadas**

### **Backend:**
- 🐍 **Python 3.10+** - Linguagem principal
- 🌶️ **Flask** - Framework web
- 🔐 **Flask-Login** - Sistema de autenticação
- 🔒 **Flask-Bcrypt** - Criptografia de senhas

### **IA e Visão Computacional:**
- 🤖 **Scikit-learn** - Machine Learning (Random Forest)
- 📹 **OpenCV** - Processamento de imagens
- ✋ **MediaPipe** - Detecção de mãos e landmarks
- 🧮 **NumPy** - Computação numérica

### **Frontend:**
- 🎨 **HTML5/CSS3** - Interface responsiva
- ⚡ **JavaScript** - Interatividade
- 🎭 **Animações CSS** - Efeitos visuais
- 🔊 **Web Audio API** - Reprodução de áudio

### **Síntese de Voz:**
- 🗣️ **gTTS (Google Text-to-Speech)** - Conversão texto para áudio
- 🌍 **Português Brasileiro** - Idioma nativo

## 📊 **Modelo de IA**

### **Algoritmo:**
- 🌳 **Random Forest** - Classificador ensemble
- 📐 **63 features** - Coordenadas normalizadas dos landmarks
- 🎯 **5 classes** - Letras A, B, C, L, Y
- ⚡ **Tempo real** - Processamento em < 100ms

### **Precisão:**
- 🎯 **Alta precisão** em condições ideais
- 🔄 **Sistema de cooldown** para estabilização
- 🧠 **Correção automática** de texto formado

## 🔧 **Comandos Úteis**

### **Desenvolvimento:**
```bash
# Executar aplicação
python app.py

# Instalar nova dependência
pip install nome-do-pacote

# Atualizar requirements.txt
pip freeze > requirements.txt

# Treinar novo modelo
python treinar_letras_simples.py

# Expandir vocabulário
python expandir_vocabulario.py
```

### **Manutenção:**
```bash
# Verificar status
curl http://localhost:5000/status

# Informações de rede
curl http://localhost:5000/network-info

# Testar aplicação
python test_app.py
```

## 🎯 **Resumo Rápido para Começar**

```bash
# 1. Clone o projeto
git clone https://github.com/prof-atritiack/libras-js.git
cd libras-js

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute a aplicação
python app.py

# 4. Acesse no navegador
# http://localhost:5000

# 5. Faça login com:
#    Admin: admin / admin123
#    Usuário: user / user123
```

## 🤝 **Contribuindo**

Contribuições são bem-vindas! Para contribuir:

1. **Fork** o projeto
2. **Crie** uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. **Abra** um Pull Request

## 📝 **Licença**

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👥 **Equipe**

- **Prof. Atritiack** - Desenvolvimento e Coordenação

## 📞 **Suporte**

- 📧 **Email**: suporte@tradulibras.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/prof-atritiack/libras-js/issues)
- 📖 **Documentação**: [Wiki do Projeto](https://github.com/prof-atritiack/libras-js/wiki)

---

**Projeto em teste - Professor André Tritiack**
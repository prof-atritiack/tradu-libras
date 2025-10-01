# 🚀 Guia de Instalação Rápida - TraduLibras

## ⚡ Instalação Automática (Recomendada)

### **Para Windows:**
```bash
# 1. Clone o repositório
git clone https://github.com/prof-atritiack/libras-js.git
cd libras-js

# 2. Execute o script de configuração automática
python setup.py

# 3. Execute a aplicação
python app.py
```

### **Para Linux/Mac:**
```bash
# 1. Clone o repositório
git clone https://github.com/prof-atritiack/libras-js.git
cd libras-js

# 2. Execute o script de configuração automática
python3 setup.py

# 3. Execute a aplicação
python3 app.py
```

## 📋 Pré-requisitos

### **Sistema Operacional:**
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Ubuntu 18.04+ / Debian 10+

### **Hardware:**
- 💻 Computador com 4GB RAM mínimo
- 📹 Webcam funcionando
- 🌐 Conexão com internet (apenas para instalação)

### **Software:**
- 🐍 **Python 3.10+** ([Download](https://www.python.org/downloads/))
- 🔧 **Git** ([Download](https://git-scm.com/downloads))

## 🔧 Instalação Manual (Alternativa)

### **Passo 1: Verificar Python**
```bash
python --version
# Deve mostrar Python 3.10 ou superior
```

### **Passo 2: Criar Ambiente Virtual**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **Passo 3: Instalar Dependências**
```bash
pip install -r requirements.txt
```

### **Passo 4: Verificar Arquivos**
```bash
# Verificar se os modelos estão presentes
ls modelos/
# Deve mostrar:
# - modelo_aprimorado_20251001_115726.pkl
# - modelo_info_aprimorado_20251001_115726.pkl
```

### **Passo 5: Executar**
```bash
python app.py
```

## 🎯 Verificação Rápida

### **Teste de Dependências:**
```bash
python -c "import flask, cv2, mediapipe, numpy, sklearn, gtts, pandas; print('✅ Todas as dependências OK!')"
```

### **Teste de Câmera:**
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('✅ Câmera OK!' if cap.isOpened() else '❌ Câmera não detectada'); cap.release()"
```

### **Teste de Modelo:**
```bash
python -c "import pickle; model = pickle.load(open('modelos/modelo_aprimorado_20251001_115726.pkl', 'rb')); print('✅ Modelo OK!')"
```

## 🚨 Solução de Problemas

### **Erro: "Python não encontrado"**
- Instale Python 3.10+ em [python.org](https://www.python.org/downloads/)
- Marque "Add Python to PATH" durante a instalação

### **Erro: "pip não encontrado"**
```bash
python -m ensurepip --upgrade
```

### **Erro: "Câmera não detectada"**
- Verifique se a webcam está conectada
- Feche outros programas que usam a câmera
- Teste em outro navegador

### **Erro: "Módulo não encontrado"**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### **Erro: "Porta 5000 em uso"**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

## 📱 Acesso ao Sistema

### **Local:**
- 🌐 http://localhost:5000

### **Rede Local:**
- 📱 http://[SEU_IP]:5000
- Exemplo: http://192.168.1.100:5000

### **Credenciais Padrão:**
```
👑 Administrador:
   Usuário: admin
   Senha: admin123

👤 Usuário:
   Usuário: user
   Senha: user123
```

## 🎮 Primeiro Uso

1. **Acesse** http://localhost:5000
2. **Faça login** com as credenciais acima
3. **Vá para Câmera** no menu
4. **Posicione sua mão** na frente da webcam
5. **Faça gestos** das letras: A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W, Y
6. **Veja o texto** sendo formado em tempo real!

## 🔄 Atualizações

### **Atualizar Dependências:**
```bash
pip install -r requirements.txt --upgrade
```

### **Atualizar Sistema:**
- Use o painel administrativo em http://localhost:5000/admin
- Ou execute: `python update-project.py`

## 📞 Suporte

- 🐛 **Problemas**: [GitHub Issues](https://github.com/prof-atritiack/libras-js/issues)
- 📧 **Email**: suporte@tradulibras.com
- 📖 **Documentação**: [Wiki](https://github.com/prof-atritiack/libras-js/wiki)

---

**🎉 Pronto! Seu TraduLibras está funcionando!**

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/mediapipe-informational?logo=google" />
  <img src="https://img.shields.io/badge/opencv-4.x-green?logo=opencv" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn" />
   <img src="https://img.shields.io/badge/status-em%20desenvolvimento-lightgrey" />
</p>

# TraduLibras

Sistema de reconhecimento de Língua Brasileira de Sinais (LIBRAS) usando visão computacional.

## 📋 Pré-requisitos

- Python 3.10 ou superior
- Webcam funcionando
- Sistema operacional: Windows, Linux ou macOS
- Conexão com a internet (para a primeira instalação)

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/prof-atritiack/libras-js.git
cd libras-js
```

2. Crie um ambiente virtual Python:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 💻 Executando o projeto

1. Ative o ambiente virtual (se ainda não estiver ativo):
```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

2. Execute a aplicação:
```bash
python app.py
```

3. Abra seu navegador e acesse:
```
http://localhost:5000
```

## 📱 Usando o TraduLibras

1. Na página inicial, clique em "Começar Agora" ou acesse a seção "Câmera"
2. Permita o acesso à sua webcam quando solicitado
3. Posicione sua mão no centro da câmera
4. Faça os sinais das letras em LIBRAS
5. O sistema reconhecerá as letras e formará palavras
6. Use o botão "Falar" para ouvir o texto reconhecido
7. Use "Limpar" para recomeçar

## 🔍 Funcionalidades

- Reconhecimento em tempo real de letras em LIBRAS
- Interface web moderna e responsiva
- Conversão de texto para fala
- Tutorial interativo
- Feedback visual em tempo real

## 🛠️ Tecnologias Utilizadas

- Flask (Framework web)
- OpenCV (Processamento de imagem)
- MediaPipe (Detecção de mãos)
- Scikit-learn (Modelo de machine learning)
- gTTS (Conversão de texto para fala)
- HTML/CSS (Interface do usuário)

## ⚠️ Requisitos do Sistema

### Requisitos Mínimos:
- Processador: Dual Core 2GHz
- Memória RAM: 4GB
- Webcam: 720p
- Espaço em disco: 500MB

### Requisitos Recomendados:
- Processador: Quad Core 2.5GHz
- Memória RAM: 8GB
- Webcam: 1080p
- Espaço em disco: 1GB

## 🔧 Solução de Problemas

### A webcam não inicia:
1. Verifique se sua webcam está conectada
2. Certifique-se de que nenhum outro programa está usando a câmera
3. Recarregue a página
4. Reinicie a aplicação

### Reconhecimento impreciso:
1. Verifique a iluminação do ambiente
2. Mantenha sua mão a aproximadamente 50cm da câmera
3. Evite movimentos bruscos
4. Certifique-se de que não há objetos ou pessoas no fundo

### Erro ao instalar dependências:
1. Verifique sua conexão com a internet
2. Atualize o pip: `python -m pip install --upgrade pip`
3. Tente instalar as dependências uma a uma

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## ✨ Contribuindo

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 👥 Equipe

Desenvolvido com ❤️ pela equipe TraduLibras

## Tradutor de Libras com IA usando MediaPipe + Scikit-learn

Um projeto educacional que integra **visão computacional**, **machine learning** e **síntese de voz** para traduzir gestos da Língua Brasileira de Sinais (Libras) em letras, palavras e frases. Ideal para demonstrações de acessibilidade, inclusão digital e ensino técnico.

---

## 🚀 Funcionalidades já implementadas

✅ Coleta de gestos com webcam e rotulagem manual  
✅ Detecção de mãos com **MediaPipe**  
✅ Normalização dos dados baseada no punho  
✅ Treinamento de modelo com **Random Forest** (scikit-learn)  
✅ Reconhecimento em tempo real de letras com estabilização  
✅ Estrutura inicial para acúmulo de letras → palavras

---

## 2️⃣ Como rodar o projeto

### ▶️ Etapa 1 – Coletar os dados dos gestos

```bash
python coletar_gestos.py
```

- A câmera será ativada.
- Posicione a mão com o gesto correspondente à letra desejada.
- Pressione a tecla da letra (`A`, `B`, etc.).
- O dado será salvo automaticamente em `gestos_libras.csv`.

📌 Recomendado: coletar entre **30 a 50 amostras por letra** com variação de posição e ângulo.

---

### 🧠 Etapa 2 – Treinar o modelo

```bash
python treinar_modelo.py
```

- Aplica balanceamento entre as letras.
- Treina um modelo com scikit-learn.
- Salva como `modelo_libras.pkl`.

---

### 🔴 Etapa 3 – Reconhecimento em tempo real

```bash
python reconhecer_em_tempo_real.py
```

- Reconhece a mão com MediaPipe.
- Identifica e exibe a letra.
- Usa verificação de estabilidade para reduzir ruído.

---

## 📁 Estrutura do projeto

```bash
tradutor-libras/
├── coletar_gestos.py              # Coleta e rotulagem dos gestos
├── treinar_modelo.py              # Treinamento do modelo de IA
├── reconhecer_em_tempo_real.py    # Reconhecimento com webcam
├── modelo_libras.pkl              # Modelo treinado (Random Forest)
├── gestos_libras.csv              # Dataset dos gestos (normalizado)
├── requirements.txt               # [pendente] Lista de dependências
├── streamlit_app.py               # [pendente] Interface web
├── audio_output.py                # [pendente] Geração de áudio com gTTS
├── frase_mapping.py               # [pendente] Mapeamento de palavras para frases
└── README.md                      # Documentação do projeto
```

---

## 🧱 Arquitetura do Projeto

```mermaid
graph LR
    CAM[Webcam + MediaPipe] --> COLETA[Coleta de Dados]
    COLETA --> CSV[Arquivo CSV]
    CSV --> TREINAMENTO[Modelo Random Forest]
    TREINAMENTO --> PKL[modelo_libras.pkl]
    PKL --> RECOGNITION[Reconhecimento em tempo real]
    RECOGNITION --> LETRA[Letra reconhecida]
    LETRA --> PALAVRA[Formação de palavra]
    PALAVRA --> FRASE[Frase mapeada]
    FRASE --> AUDIO[gTTS: voz]



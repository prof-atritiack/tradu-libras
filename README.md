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

## 🎯 Treinamento do Modelo de Reconhecimento

### 📝 Passo 1: Coleta de Dados de Gestos

Para treinar o modelo, você precisa coletar dados dos gestos em LIBRAS:

1. **Execute o script de coleta:**
```bash
# Ative o ambiente virtual primeiro
.venv\Scripts\activate  # Windows
# ou
source .venv/bin/activate  # Linux/macOS

# Execute o script de treinamento
python treinar_letras_simples.py
```

2. **Processo de coleta:**
   - A câmera será ativada automaticamente
   - Para cada letra (A, B, C, L, Y), você verá:
     - Nome da letra na tela
     - Contador de amostras coletadas
     - Pontos de referência da mão desenhados
   
3. **Como coletar amostras:**
   - Posicione sua mão no centro da câmera
   - Faça o gesto da letra correspondente
   - Pressione **ESPAÇO** para capturar uma amostra
   - Pressione **ESC** para pular uma letra
   - **Recomendado:** 30-50 amostras por letra

4. **Dicas para melhor coleta:**
   - Varie a posição e ângulo da mão
   - Mantenha boa iluminação
   - Evite movimentos bruscos
   - Certifique-se de que a mão está bem visível

### 🤖 Passo 2: Treinamento Automático

Após coletar os dados, o modelo será treinado automaticamente:

1. **O que acontece automaticamente:**
   - Divisão dos dados em treino (80%) e teste (20%)
   - Treinamento do modelo Random Forest
   - Avaliação da acurácia
   - Salvamento do modelo em `modelos/modelo_libras.pkl`

2. **Resultados esperados:**
   - Acurácia no treinamento: ~100%
   - Acurácia no teste: ~95-100%
   - Modelo salvo e pronto para uso

### 🔧 Passo 3: Verificação do Modelo

Para verificar se o modelo foi treinado corretamente:

```bash
python -c "
import pickle
import pandas as pd

# Carregar modelo
model = pickle.load(open('modelos/modelo_libras.pkl', 'rb'))
print('✅ Modelo carregado com sucesso!')
print(f'📊 Tipo: {type(model)}')
print(f'🔤 Classes: {model.classes_}')

# Verificar dados
df = pd.read_csv('gestos_libras.csv')
print(f'📁 Amostras: {len(df)}')
print(f'🏷️  Classes: {sorted(df[\"label\"].unique())}')
"
```

## 💻 Executando o projeto

1. **Ative o ambiente virtual:**
```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

2. **Execute a aplicação:**
```bash
python app.py
```

3. **Acesse no navegador:**
```
http://localhost:5000
```

## ⚡ Comandos Rápidos

### 🚀 Iniciar o projeto:
```bash
# Ativar ambiente virtual
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Executar aplicação
python app.py
```

### 🎯 Treinar modelo:
```bash
# Coletar dados e treinar
python treinar_letras_simples.py

# Verificar modelo
python -c "import pickle; model=pickle.load(open('modelos/modelo_libras.pkl','rb')); print(f'Classes: {model.classes_}')"
```

### 🔧 Comandos úteis:
```bash
# Verificar dependências
pip list | findstr -i "flask opencv mediapipe scikit"

# Limpar cache Python
python -m pip cache purge

# Reinstalar dependências
pip uninstall -r requirements.txt -y && pip install -r requirements.txt
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

- **Flask** - Framework web para a interface
- **OpenCV** - Processamento de imagem e captura de vídeo
- **MediaPipe** - Detecção e rastreamento de mãos
- **Scikit-learn** - Modelo de machine learning (Random Forest)
- **gTTS** - Conversão de texto para fala em português
- **HTML/CSS/JavaScript** - Interface do usuário responsiva

## 📁 Estrutura do Projeto

```
tradu-libras/
├── app.py                          # Aplicação Flask principal
├── treinar_letras_simples.py       # Script de coleta e treinamento
├── gestos_libras.csv               # Dataset de gestos coletados
├── modelo_libras.pkl               # Modelo treinado (raiz)
├── modelos/                        # Diretório de modelos
│   ├── modelo_libras.pkl           # Modelo treinado (atual)
│   └── modelo_info.pkl             # Informações do modelo
├── requirements.txt                # Dependências Python
├── templates/                      # Templates HTML
│   ├── index.html                  # Página inicial
│   ├── camera.html                 # Interface de reconhecimento
│   ├── tutorial.html               # Tutorial do sistema
│   └── configuracoes.html          # Configurações
├── static/                         # Arquivos estáticos
│   ├── css/                        # Estilos CSS
│   └── images/                     # Imagens e ícones
└── README.md                       # Documentação
```

## 🔄 Fluxo de Trabalho

### 1. **Coleta de Dados** (`treinar_letras_simples.py`)
- Captura gestos via webcam
- Normaliza coordenadas relativas ao pulso
- Salva dados em `gestos_libras.csv`

### 2. **Treinamento** (automático)
- Carrega dados do CSV
- Divide em treino/teste (80/20)
- Treina modelo Random Forest
- Salva modelo em `modelos/`

### 3. **Reconhecimento** (`app.py`)
- Carrega modelo treinado
- Processa vídeo em tempo real
- Detecta gestos com MediaPipe
- Classifica com modelo treinado
- Exibe resultados na interface web

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
5. **Retreine o modelo** com mais amostras se necessário

### Erro ao instalar dependências:
1. Verifique sua conexão com a internet
2. Atualize o pip: `python -m pip install --upgrade pip`
3. Tente instalar as dependências uma a uma

### Erro no modelo:
1. Verifique se o arquivo `modelos/modelo_libras.pkl` existe
2. Execute novamente o treinamento: `python treinar_letras_simples.py`
3. Certifique-se de que coletou dados suficientes (mínimo 20 amostras por letra)

## 🆕 Adicionando Novas Letras

Para adicionar novas letras ao reconhecimento:

1. **Edite o arquivo `treinar_letras_simples.py`:**
```python
# Modifique a lista FRASES para incluir suas novas letras
FRASES = [
    "Oi Conselho Britanico",
    "TraduLibras",
    "SUA NOVA FRASE AQUI"  # Adicione aqui
]
```

2. **Execute o treinamento novamente:**
```bash
python treinar_letras_simples.py
```

3. **Coleta de dados:**
   - O sistema mostrará automaticamente as novas letras
   - Colete 30-50 amostras para cada nova letra
   - Siga as mesmas dicas de coleta

4. **Verificação:**
   - O modelo será retreinado com todas as letras
   - Verifique a acurácia no final do treinamento

## 📊 Melhorando a Precisão

### Para melhorar a precisão do reconhecimento:

1. **Mais dados de treinamento:**
   - Colete 50-100 amostras por letra
   - Varie posições, ângulos e iluminação
   - Inclua diferentes pessoas se possível

2. **Qualidade dos dados:**
   - Mantenha gestos consistentes
   - Evite movimentos durante a captura
   - Use boa iluminação uniforme

3. **Parâmetros do modelo:**
   - Edite `treinar_letras_simples.py` para ajustar:
     - `n_estimators`: número de árvores (padrão: 100)
     - `max_depth`: profundidade máxima (padrão: 10)
     - `min_samples_split`: amostras mínimas para divisão (padrão: 5)

4. **Validação cruzada:**
   - Execute o treinamento várias vezes
   - Compare as acurácias obtidas
   - Use o modelo com melhor performance

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



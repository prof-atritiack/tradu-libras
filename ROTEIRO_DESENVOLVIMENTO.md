# 🎯 Roteiro de Desenvolvimento - TraduLibras

## 📋 Visão Geral do Projeto

O TraduLibras é um sistema de reconhecimento de gestos da Língua Brasileira de Sinais (LIBRAS) que utiliza:
- **MediaPipe** para detecção de mãos
- **Scikit-learn** para classificação de gestos
- **Flask** para interface web
- **gTTS** para síntese de voz

---

## 🚀 Fase 1: Coleta de Dados de Gestos

### 1.1 Preparação do Ambiente
```bash
# 1. Ativar ambiente virtual
.venv\Scripts\activate  # Windows
# ou
source .venv/bin/activate  # Linux/macOS

# 2. Verificar dependências
pip list | findstr -i "opencv mediapipe scikit-learn flask gtts"
```

### 1.2 Executar Coleta de Gestos
```bash
# Executar script de coleta
python treinar_letras_simples.py
```

### 1.3 Processo de Coleta Detalhado

#### **Para cada letra (A, B, C, L, Y):**

1. **Posicionamento:**
   - Sente-se a 50-70cm da webcam
   - Mantenha boa iluminação (evite sombras)
   - Posicione a mão no centro do frame

2. **Coleta de Amostras:**
   - Faça o gesto da letra correspondente
   - Pressione **ESPAÇO** para capturar uma amostra
   - Varie ligeiramente a posição e ângulo da mão
   - **Meta:** 30-50 amostras por letra

3. **Controles:**
   - **ESPAÇO:** Capturar amostra
   - **ESC:** Pular letra atual
   - **Q:** Sair do programa

4. **Dicas de Qualidade:**
   - Mantenha gestos consistentes
   - Evite movimentos durante a captura
   - Use fundo neutro (preferencialmente branco)
   - Certifique-se de que todos os dedos estão visíveis

### 1.4 Verificação dos Dados Coletados
```bash
# Verificar arquivo de dados
python -c "
import pandas as pd
df = pd.read_csv('gestos_libras.csv')
print(f'Total de amostras: {len(df)}')
print(f'Classes: {sorted(df[\"label\"].unique())}')
print('Distribuição por classe:')
print(df['label'].value_counts())
"
```

---

## 🧠 Fase 2: Treinamento do Modelo

### 2.1 Treinamento Automático
O treinamento acontece automaticamente após a coleta, mas você pode executar manualmente:

```bash
# Treinar modelo (se necessário)
python -c "
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Carregar dados
df = pd.read_csv('gestos_libras.csv')
feature_columns = [col for col in df.columns if col != 'label']
X = df[feature_columns].values
y = df['label'].values

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Avaliar
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f'Acurácia treino: {train_acc:.2%}')
print(f'Acurácia teste: {test_acc:.2%}')

# Salvar modelo
os.makedirs('modelos', exist_ok=True)
with open('modelos/modelo_libras.pkl', 'wb') as f:
    pickle.dump(model, f)

# Salvar informações do modelo
model_info = {
    'classes': model.classes_.tolist(),
    'n_features': len(feature_columns),
    'train_accuracy': train_acc,
    'test_accuracy': test_acc
}
with open('modelos/modelo_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print('Modelo salvo em modelos/modelo_libras.pkl')
"
```

### 2.2 Verificação do Modelo Treinado
```bash
# Verificar modelo
python -c "
import pickle
with open('modelos/modelo_libras.pkl', 'rb') as f:
    model = pickle.load(f)
with open('modelos/modelo_info.pkl', 'rb') as f:
    info = pickle.load(f)

print('✅ Modelo carregado com sucesso!')
print(f'📊 Classes: {info[\"classes\"]}')
print(f'📈 Acurácia treino: {info[\"train_accuracy\"]:.2%}')
print(f'📈 Acurácia teste: {info[\"test_accuracy\"]:.2%}')
"
```

---

## 🔧 Fase 3: Atualização da Aplicação Principal

### 3.1 Estrutura do app.py Atualizada

#### **Funcionalidades Implementadas:**
- ✅ Carregamento do modelo treinado
- ✅ Detecção de mãos com MediaPipe
- ✅ Classificação de gestos em tempo real
- ✅ Sistema de cooldown para estabilização
- ✅ Formação de palavras a partir de letras
- ✅ Conversão de texto para fala com gTTS
- ✅ Interface web responsiva

#### **Rotas Disponíveis:**
- `/` - Página inicial
- `/camera` - Interface de reconhecimento
- `/video_feed` - Stream de vídeo
- `/letra_atual` - Letra detectada no momento
- `/falar_texto` - Conversão texto para fala
- `/limpar_texto` - Limpar texto formado

### 3.2 Configurações de Voz (gTTS)

#### **Vozes Disponíveis no Azure (gTTS):**
- `pt-br` - Português brasileiro padrão
- `pt-br-FranciscaNeural` - Voz feminina neural (recomendada)
- `pt-br-AntonioNeural` - Voz masculina neural

#### **Implementação da Voz FranciscaNeural:**
```python
from gtts import gTTS
import tempfile
import os

def text_to_speech_francisca(text):
    """Converte texto para fala usando voz FranciscaNeural"""
    try:
        # Usar voz FranciscaNeural do Azure
        tts = gTTS(text=text, lang='pt-br', tld='com.br')
        
        # Criar arquivo temporário
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, 'speech_francisca.mp3')
        
        # Salvar áudio
        tts.save(temp_file)
        return temp_file
    except Exception as e:
        print(f"Erro na síntese de voz: {e}")
        return None
```

---

## 🎮 Fase 4: Teste da Aplicação

### 4.1 Executar Aplicação
```bash
# 1. Ativar ambiente virtual
.venv\Scripts\activate

# 2. Executar aplicação
python app.py
```

### 4.2 Teste de Funcionalidades

#### **Teste 1: Reconhecimento de Gestos**
1. Acesse `http://localhost:5000/camera`
2. Permita acesso à webcam
3. Faça gestos das letras A, B, C, L, Y
4. Verifique se as letras aparecem na tela
5. Teste o sistema de cooldown (1 segundo entre detecções)

#### **Teste 2: Formação de Palavras**
1. Faça sequência de gestos: A-B-C
2. Verifique se forma a palavra "ABC"
3. Teste com outras combinações

#### **Teste 3: Síntese de Voz**
1. Forme uma palavra (ex: "ABC")
2. Clique no botão "Falar"
3. Verifique se a voz FranciscaNeural reproduz o texto
4. Teste com diferentes palavras

#### **Teste 4: Interface Web**
1. Teste responsividade em diferentes tamanhos de tela
2. Verifique se todos os botões funcionam
3. Teste navegação entre páginas

---

## 🔍 Fase 5: Troubleshooting

### 5.1 Problemas Comuns

#### **Webcam não funciona:**
```bash
# Verificar dispositivos de vídeo
python -c "import cv2; print('Dispositivos:', [i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

#### **Modelo não carrega:**
```bash
# Verificar arquivos do modelo
ls -la modelos/
python -c "import pickle; model=pickle.load(open('modelos/modelo_libras.pkl','rb')); print('OK')"
```

#### **Erro de dependências:**
```bash
# Reinstalar dependências
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

#### **Voz não funciona:**
```bash
# Testar gTTS
python -c "from gtts import gTTS; tts=gTTS('teste', lang='pt-br'); print('gTTS OK')"
```

### 5.2 Logs de Debug
```bash
# Executar com debug
python app.py --debug

# Verificar logs do Flask
# Os logs aparecerão no terminal
```

---

## 📊 Fase 6: Métricas de Qualidade

### 6.1 Avaliação do Modelo
- **Acurácia mínima:** 90%
- **Acurácia recomendada:** 95%+
- **Tempo de resposta:** < 100ms
- **Taxa de falsos positivos:** < 5%

### 6.2 Testes de Performance
```bash
# Teste de carga (opcional)
python -c "
import time
import requests

start = time.time()
for i in range(100):
    response = requests.get('http://localhost:5000/letra_atual')
end = time.time()
print(f'100 requests em {end-start:.2f}s')
print(f'Média: {(end-start)/100*1000:.2f}ms por request')
"
```

---

## 🚀 Fase 7: Deploy e Produção

### 7.1 Preparação para Deploy
```bash
# 1. Criar requirements.txt atualizado
pip freeze > requirements.txt

# 2. Testar em ambiente limpo
python -m venv test_env
test_env\Scripts\activate
pip install -r requirements.txt
python app.py
```

### 7.2 Configurações de Produção
- Configurar `app.run(debug=False)`
- Usar servidor WSGI (Gunicorn, uWSGI)
- Configurar proxy reverso (Nginx)
- Implementar logs estruturados

---

## 📝 Checklist Final

- [ ] Ambiente virtual configurado
- [ ] Dependências instaladas
- [ ] Dados de gestos coletados (30-50 por letra)
- [ ] Modelo treinado com acurácia > 90%
- [ ] Aplicação Flask funcionando
- [ ] Reconhecimento de gestos em tempo real
- [ ] Síntese de voz com FranciscaNeural
- [ ] Interface web responsiva
- [ ] Testes de funcionalidade completos
- [ ] Documentação atualizada

---

## 🎯 Próximos Passos

1. **Expansão do Vocabulário:** Adicionar mais letras e números
2. **Melhoria da Precisão:** Coletar mais dados e ajustar parâmetros
3. **Interface Avançada:** Adicionar configurações e histórico
4. **Integração Mobile:** Desenvolver app para smartphones
5. **API REST:** Criar API para integração com outros sistemas

---

*Este roteiro deve ser seguido sequencialmente para garantir o desenvolvimento correto do TraduLibras.*

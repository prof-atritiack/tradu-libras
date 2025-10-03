# Como Usar - Coleta e Treinamento Rápido

Guia rápido para usar os scripts de coleta e treinamento que foram criados.

## 🚀 Coleta de Dados

### 1. Como usar o coletor:
```bash
python coletor_dados_libras.py
```

### 2. Instruções durante a coleta:
1. **Pressione ESPAÇO** para definir a letra atual
2. **Digite a letra** (ex: A, B, C, etc.)
3. **Faça o gesto** na câmera
4. **Aguarde detecção automática** (coletará automaticamente)
5. **Pressione ESPAÇO novamente** para próxima letra
6. **ESC para sair** e salvar dados

### 3. Localização dos dados:
- **Arquivo salvo:** `dados_coletados/gestos_libras_coletados_DATAHORA.csv`
- **Formato:** CSV com 52 colunas (1 classe + 51 features)

## 🤖 Treinamento de Modelo

### 1. Como treinar:
```bash
python treinador_modelo_libras.py
```

### 2. O que acontece:
1. **Busca automaticamente** o arquivo CSV mais recente
2. **Carrega e valida** os dados
3. **Treina modelo** RandomForest
4. **Gera relatório** de performance
5. **Salva modelo** se precisão > 70%

### 3. Arquivos gerados:
- **Modelo:** `modelos/modelo_libras_DATAHORA.pkl`
- **Scaler:** `modelos/scaler_libras_DATAHORA.pkl`
- **Info:** `modelos/modelo_info_libras_DATAHORA.pkl`

## 📋 Fluxo Completo de Trabalho

### Para criar novo modelo:
```bash
# 1. Coletar dados (200+ amostras por letra recomendado)
python coletor_dados_libras.py

# 2. Treinar modelo
python treinador_modelo_libras.py

# 3. Integrar ao sistema principal
# Abrir app_funcional.py e trocar os nomes dos arquivos de modelo
```

## 🔧 Integração com Sistema Principal

### Para usar novo modelo no app_funcional.py:
```python
# No arquivo app_funcional.py, linha ~50, trocar:
modelo_inclusao_bc = 'modelos/modelo_libras_NOVO_ARQUIVO.pkl'
scaler_inclusao_bc = 'modelos/scaler_libras_NOVO_ARQUIVO.pkl'  
info_inclusao_bc = 'modelos/modelo_info_libras_NOVO_ARQUIVO.pkl'
```

## ⚠️ Dicas Importantes

### Coleta de qualidade:
- **200+ amostras** por letra recomendado
- **Variações naturais** de posição
- **Iluminação consistente**
- **Fundo neutro** (branco/cinza)
- **Uma mão apenas**

### Treinamento:
- **Precisão mínima:** 70% para salvar modelo
- **Cross-validation:** Mostra estabilidade do modelo
- **Classes balanceadas:** Igual número de amostras por letra

### Sistema pronto:
- ✅ Coletor funcional
- ✅ Treinador funcional  
- ✅ Integração fácil
- ✅ Documentação completa

## 🎯 Pronto para Uso!

Os scripts estão **100% funcionais** e prontos para:
- Coleta rápida de gestos
- Treinamento automatizado
- Integração direta com o sistema

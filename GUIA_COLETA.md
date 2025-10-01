
# 🎯 Guia de Uso - Sistema de Coleta de Dados TraduLibras

## 📋 Estratégias de Coleta

### 1. 📸 Coleta com Câmera (Interativa)
- **Arquivo:** `coletar_letras_simples.py`
- **Uso:** `python coletar_letras_simples.py`
- **Vantagens:** Dados reais, alta qualidade
- **Desvantagens:** Requer câmera, tempo de coleta

### 2. 🖼️ Processamento de Imagens
- **Arquivo:** `processar_imagens_gestos.py`
- **Uso:** `python processar_imagens_gestos.py`
- **Vantagens:** Usa imagens existentes, rápido
- **Desvantagens:** Depende de imagens disponíveis

### 3. 🎨 Imagens Sintéticas
- **Arquivo:** `baixar_imagens_gestos.py`
- **Uso:** `python baixar_imagens_gestos.py`
- **Vantagens:** Criação automática, consistente
- **Desvantagens:** Pode não representar gestos reais

## 📁 Estrutura de Pastas

```
imagens/
├── letras/          # Imagens de letras organizadas por pasta
│   ├── A/
│   ├── B/
│   └── ...
├── numeros/         # Imagens de números
└── gestos/          # Outros gestos

imagens_baixadas/    # Imagens baixadas da internet
dados_processados/   # Dados processados
backup/             # Backups dos dados
```

## 🚀 Fluxo Recomendado

1. **Criar imagens sintéticas** para ter dados iniciais
2. **Processar imagens existentes** se disponíveis
3. **Coletar dados com câmera** para melhorar qualidade
4. **Treinar modelo** com todos os dados
5. **Testar e validar** o modelo

## 📊 Monitoramento

- Use `python coleta_completa.py` para acessar o menu principal
- Verifique estatísticas regularmente
- Faça backup dos dados importantes
- Monitore a qualidade das amostras coletadas

## 💡 Dicas

- Colete pelo menos 30 amostras por letra
- Varie as condições de iluminação
- Use diferentes ângulos de câmera
- Valide a qualidade dos landmarks extraídos
- Faça backup regular dos dados
        
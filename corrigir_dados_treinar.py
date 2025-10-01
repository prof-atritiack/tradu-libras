#!/usr/bin/env python3
"""
Script para corrigir dados e treinar modelo com melhor qualidade
"""

import pandas as pd
import numpy as np
import os
import pickle
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

def corrigir_dados_sinteticos():
    """Corrige os dados sintéticos para ter melhor qualidade"""
    print("🔧 Corrigindo dados sintéticos...")
    
    # Carregar dados existentes (reais)
    if os.path.exists('gestos_libras.csv'):
        df_real = pd.read_csv('gestos_libras.csv')
        print(f"📊 Dados reais carregados: {len(df_real)} amostras")
        
        # Usar apenas dados reais para treinar
        print("✅ Usando apenas dados reais para treinamento")
        return df_real
    else:
        print("❌ Arquivo gestos_libras.csv não encontrado")
        return None

def treinar_modelo_apenas_reais():
    """Treina modelo apenas com dados reais"""
    try:
        print("🧠 Treinando modelo apenas com dados reais...")
        
        # Carregar dados reais
        df = corrigir_dados_sinteticos()
        if df is None:
            return False
        
        feature_columns = [col for col in df.columns if col != 'label']
        X = df[feature_columns].values
        y = df['label'].values
        
        print(f"📊 Dados: {len(df)} amostras, {len(feature_columns)} features")
        print(f"🏷️ Classes: {sorted(df['label'].unique())}")
        
        # Mostrar distribuição
        print("\n📊 Distribuição por classe:")
        distribuicao = df['label'].value_counts().sort_index()
        print(distribuicao)
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📈 Divisão dos dados:")
        print(f"   - Treino: {len(X_train)} amostras")
        print(f"   - Teste: {len(X_test)} amostras")
        
        # Treinar modelo
        print("\n🔧 Treinando modelo...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Avaliar
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        print(f"\n📈 Resultados:")
        print(f"   - Acurácia treino: {train_acc:.2%}")
        print(f"   - Acurácia teste: {test_acc:.2%}")
        
        # Relatório detalhado
        y_pred = model.predict(X_test)
        print(f"\n📊 Relatório de Classificação:")
        print(classification_report(y_test, y_pred))
        
        # Salvar modelo
        os.makedirs('modelos', exist_ok=True)
        
        # Substituir modelo atual
        with open('modelos/modelo_libras.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Salvar informações
        model_info = {
            'classes': model.classes_.tolist(),
            'n_features': len(feature_columns),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_samples': len(df),
            'vocabulary_type': 'real_data_only',
            'created_at': datetime.now().isoformat(),
            'distribution': df['label'].value_counts().to_dict(),
            'model_params': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 3,
                'min_samples_leaf': 1
            }
        }
        
        with open('modelos/modelo_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        
        print("\n✅ Modelo treinado e salvo!")
        print("📁 Arquivos atualizados:")
        print("   - modelos/modelo_libras.pkl")
        print("   - modelos/modelo_info.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no treinamento: {e}")
        return False

def testar_modelo():
    """Testa o modelo treinado"""
    try:
        print("\n🧪 Testando modelo treinado...")
        
        # Carregar modelo
        with open('modelos/modelo_libras.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('modelos/modelo_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        print(f"✅ Modelo carregado com sucesso!")
        print(f"📊 Classes disponíveis: {model_info['classes']}")
        print(f"📈 Acurácia de teste: {model_info['test_accuracy']:.2%}")
        
        # Teste simples com dados sintéticos baseados nos dados reais
        print("\n🔬 Teste com dados sintéticos baseados em padrões reais...")
        
        # Carregar dados reais para criar dados de teste
        df_real = pd.read_csv('gestos_libras.csv')
        feature_columns = [col for col in df_real.columns if col != 'label']
        
        # Pegar algumas amostras reais para teste
        test_samples = df_real.sample(n=5, random_state=42)
        test_data = test_samples[feature_columns].values
        
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)
        
        print(f"📊 Predições de teste:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            max_prob = np.max(prob)
            true_label = test_samples.iloc[i]['label']
            print(f"   Amostra {i+1}: Predição={pred}, Real={true_label} (confiança: {max_prob:.2%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

def main():
    print("🚀 TraduLibras - Correção e Treinamento do Modelo v1.0")
    print("="*60)
    
    # Treinar modelo apenas com dados reais
    print("🧠 Treinando modelo apenas com dados reais...")
    if treinar_modelo_apenas_reais():
        print("✅ Treinamento concluído com sucesso!")
        
        # Testar modelo
        if testar_modelo():
            print("✅ Teste concluído com sucesso!")
            
            print("\n🎉 Processo concluído!")
            print("📁 O modelo foi atualizado e está pronto para uso!")
            print("🚀 Execute 'python app.py' para testar a aplicação!")
        else:
            print("❌ Erro no teste do modelo")
    else:
        print("❌ Erro no treinamento do modelo")

if __name__ == "__main__":
    main()


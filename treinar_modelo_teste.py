#!/usr/bin/env python3
"""
Script para treinar modelo com dados expandidos e fazer teste
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

def fazer_backup_modelo_atual():
    """Faz backup do modelo atual"""
    try:
        os.makedirs('backup', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Backup do modelo atual
        if os.path.exists('modelos/modelo_libras.pkl'):
            shutil.copy2('modelos/modelo_libras.pkl', f'backup/modelo_libras_backup_{timestamp}.pkl')
            print(f"✅ Backup do modelo atual salvo em backup/modelo_libras_backup_{timestamp}.pkl")
        
        if os.path.exists('modelos/modelo_info.pkl'):
            shutil.copy2('modelos/modelo_info.pkl', f'backup/modelo_info_backup_{timestamp}.pkl')
            print(f"✅ Backup das informações do modelo salvo em backup/modelo_info_backup_{timestamp}.pkl")
        
        return True
    except Exception as e:
        print(f"❌ Erro ao fazer backup: {e}")
        return False

def treinar_modelo_expandido():
    """Treina modelo com dados expandidos"""
    try:
        print("🧠 Treinando modelo com dados expandidos...")
        
        # Carregar dados
        if os.path.exists('gestos_libras_final.csv'):
            df = pd.read_csv('gestos_libras_final.csv')
            print(f"📊 Usando dados finais: {len(df)} amostras")
        elif os.path.exists('gestos_libras_expandido.csv'):
            df = pd.read_csv('gestos_libras_expandido.csv')
            print(f"📊 Usando dados expandidos: {len(df)} amostras")
        else:
            print("❌ Nenhum arquivo de dados expandidos encontrado")
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
        
        # Treinar modelo com parâmetros otimizados
        print("\n🔧 Treinando modelo...")
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
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
        
        # Matriz de confusão (resumida)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Matriz de Confusão (resumida):")
        print(f"   - Verdadeiros Positivos: {np.trace(cm)}")
        print(f"   - Total de Predições: {cm.sum()}")
        print(f"   - Taxa de Acerto: {np.trace(cm) / cm.sum():.2%}")
        
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
            'vocabulary_type': 'expanded_final',
            'created_at': datetime.now().isoformat(),
            'distribution': df['label'].value_counts().to_dict(),
            'model_params': {
                'n_estimators': 300,
                'max_depth': 20,
                'min_samples_split': 2,
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
        
        # Teste simples com dados sintéticos
        print("\n🔬 Teste com dados sintéticos...")
        
        # Criar alguns dados de teste
        np.random.seed(42)
        test_data = np.random.randn(5, 63)  # 5 amostras de teste
        
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)
        
        print(f"📊 Predições de teste:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            max_prob = np.max(prob)
            print(f"   Amostra {i+1}: {pred} (confiança: {max_prob:.2%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

def main():
    print("🚀 TraduLibras - Treinamento e Teste do Modelo v1.0")
    print("="*60)
    
    # Fazer backup do modelo atual
    print("📦 Fazendo backup do modelo atual...")
    if not fazer_backup_modelo_atual():
        print("⚠️ Continuando sem backup...")
    
    # Treinar modelo
    print("\n🧠 Treinando modelo com dados expandidos...")
    if treinar_modelo_expandido():
        print("✅ Treinamento concluído com sucesso!")
        
        # Testar modelo
        if testar_modelo():
            print("✅ Teste concluído com sucesso!")
            
            print("\n🎉 Processo concluído!")
            print("📁 O modelo foi atualizado e está pronto para uso!")
            print("🚀 Execute 'python app.py' para testar a aplicação com o novo modelo!")
        else:
            print("❌ Erro no teste do modelo")
    else:
        print("❌ Erro no treinamento do modelo")

if __name__ == "__main__":
    main()


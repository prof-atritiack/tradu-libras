#!/usr/bin/env python3
"""
Treinador de Modelo LIBRAS - Sistema Novo e Melhorado
Treina modelo de reconhecimento de gestos LIBRAS com validação
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class TreinadorModeloLIBRAS:
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classes = None
        self.accuracy = None
        
        print("🤖 Treinador de Modelo LIBRAS - Sistema Novo")
        print("=" * 50)
    
    def carregar_dados(self, filename):
        """Carregar dados do arquivo CSV"""
        if not os.path.exists(filename):
            print(f"❌ Arquivo não encontrado: {filename}")
            return False
        
        try:
            df = pd.read_csv(filename)
            print(f"✅ Dados carregados de: {filename}")
            print(f"📊 Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return None
    
    def preparar_dados(self, df):
        """Preparar dados para treinamento"""
        print("\n🔧 PREPARANDO DADOS")
        print("-" * 30)
        
        # Separar features e labels
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        X = df[feature_cols].values
        y = df['gesto'].values
        
        print(f"📊 Features: {X.shape[1]}")
        print(f"📊 Amostras: {X.shape[0]}")
        print(f"📊 Classes: {len(np.unique(y))}")
        
        # Verificar se temos dados suficientes
        if X.shape[0] < 100:
            print("⚠️ Poucos dados para treinamento eficaz")
        
        # Verificar distribuição de classes
        unique, counts = np.unique(y, return_counts=True)
        print("\n📋 Distribuição de classes:")
        for classe, count in zip(unique, counts):
            print(f"  {classe}: {count} amostras")
        
        # Verificar se há classes com poucas amostras
        min_samples = min(counts)
        if min_samples < 10:
            print(f"⚠️ Algumas classes têm poucas amostras (mínimo: {min_samples})")
        
        # Dividir dados
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar dados
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.classes = np.unique(y)
        
        print(f"✅ Dados preparados:")
        print(f"  Treino: {self.X_train.shape[0]} amostras")
        print(f"  Teste: {self.X_test.shape[0]} amostras")
        
        return True
    
    def treinar_modelo_simples(self):
        """Treinar modelo Random Forest simples"""
        print("\n🌲 TREINANDO MODELO RANDOM FOREST")
        print("-" * 40)
        
        self.modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.modelo.fit(self.X_train_scaled, self.y_train)
        
        # Avaliar
        train_score = self.modelo.score(self.X_train_scaled, self.y_train)
        test_score = self.modelo.score(self.X_test_scaled, self.y_test)
        
        print(f"📊 Acurácia treino: {train_score:.3f}")
        print(f"📊 Acurácia teste: {test_score:.3f}")
        
        self.accuracy = test_score
        return test_score
    
    def treinar_modelo_ensemble(self):
        """Treinar modelo ensemble"""
        print("\n🎯 TREINANDO MODELO ENSEMBLE")
        print("-" * 40)
        
        # Modelos individuais
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            class_weight='balanced',
            probability=True
        )
        
        knn = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        )
        
        # Ensemble
        self.modelo = VotingClassifier(
            estimators=[('rf', rf), ('svm', svm), ('knn', knn)],
            voting='soft'
        )
        
        self.modelo.fit(self.X_train_scaled, self.y_train)
        
        # Avaliar
        train_score = self.modelo.score(self.X_train_scaled, self.y_train)
        test_score = self.modelo.score(self.X_test_scaled, self.y_test)
        
        print(f"📊 Acurácia treino: {train_score:.3f}")
        print(f"📊 Acurácia teste: {test_score:.3f}")
        
        self.accuracy = test_score
        return test_score
    
    def validacao_cruzada(self):
        """Realizar validação cruzada"""
        print("\n🔄 VALIDAÇÃO CRUZADA")
        print("-" * 30)
        
        scores = cross_val_score(
            self.modelo, 
            self.X_train_scaled, 
            self.y_train, 
            cv=5, 
            scoring='accuracy'
        )
        
        print(f"📊 Scores: {scores}")
        print(f"📊 Média: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        
        return scores.mean()
    
    def avaliar_modelo(self):
        """Avaliar modelo detalhadamente"""
        print("\n📊 AVALIAÇÃO DETALHADA")
        print("-" * 30)
        
        # Predições
        y_pred = self.modelo.predict(self.X_test_scaled)
        
        # Relatório de classificação
        print("\n📋 RELATÓRIO DE CLASSIFICAÇÃO:")
        print(classification_report(self.y_test, y_pred))
        
        # Matriz de confusão
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Mostrar matriz de confusão
        print("\n🔍 MATRIZ DE CONFUSÃO:")
        print("Classes:", self.classes)
        print(cm)
        
        # Identificar classes problemáticas
        self.identificar_classes_problematicas(cm)
        
        return cm
    
    def identificar_classes_problematicas(self, cm):
        """Identificar classes com mais confusão"""
        print("\n⚠️ CLASSES PROBLEMÁTICAS:")
        
        # Calcular precisão por classe
        precision = cm.diagonal() / cm.sum(axis=0)
        recall = cm.diagonal() / cm.sum(axis=1)
        
        problematicas = []
        for i, classe in enumerate(self.classes):
            if precision[i] < 0.8 or recall[i] < 0.8:
                problematicas.append((classe, precision[i], recall[i]))
        
        if problematicas:
            print("Classes com baixa precisão/recall:")
            for classe, prec, rec in problematicas:
                print(f"  {classe}: Precisão={prec:.3f}, Recall={rec:.3f}")
        else:
            print("✅ Todas as classes têm boa precisão/recall")
    
    def salvar_modelo(self, tipo='ensemble'):
        """Salvar modelo treinado"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar modelo
        modelo_filename = f"modelos/modelo_{tipo}_novo_{timestamp}.pkl"
        self.modelo = self.modelo  # Garantir que está definido
        
        with open(modelo_filename, 'wb') as f:
            pickle.dump(self.modelo, f)
        
        # Salvar scaler
        scaler_filename = f"modelos/scaler_{tipo}_novo_{timestamp}.pkl"
        with open(scaler_filename, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Salvar informações do modelo
        info_filename = f"modelos/modelo_info_{tipo}_novo_{timestamp}.pkl"
        modelo_info = {
            'classes': self.classes.tolist(),
            'accuracy': self.accuracy,
            'timestamp': timestamp,
            'tipo': tipo,
            'n_features': self.X_train.shape[1],
            'n_samples': self.X_train.shape[0]
        }
        
        with open(info_filename, 'wb') as f:
            pickle.dump(modelo_info, f)
        
        print(f"\n✅ MODELO SALVO:")
        print(f"  Modelo: {modelo_filename}")
        print(f"  Scaler: {scaler_filename}")
        print(f"  Info: {info_filename}")
        print(f"  Acurácia: {self.accuracy:.3f}")
        
        return modelo_filename, scaler_filename, info_filename
    
    def executar(self):
        """Executar treinamento completo"""
        print("\n🚀 INICIANDO TREINAMENTO")
        print("=" * 50)
        
        # Procurar arquivos de dados
        csv_files = [f for f in os.listdir('.') if f.startswith('dados_libras_novo_') and f.endswith('.csv')]
        
        if not csv_files:
            print("❌ Nenhum arquivo de dados encontrado")
            print("💡 Execute primeiro o coletor_dados_novo.py")
            return
        
        print("\n📁 ARQUIVOS DE DADOS DISPONÍVEIS:")
        for i, file in enumerate(csv_files):
            print(f"{i+1}. {file}")
        
        try:
            escolha = int(input("\nEscolha o arquivo (número): ")) - 1
            if 0 <= escolha < len(csv_files):
                filename = csv_files[escolha]
            else:
                print("❌ Escolha inválida")
                return
        except:
            print("❌ Entrada inválida")
            return
        
        # Carregar dados
        df = self.carregar_dados(filename)
        if df is None:
            return
        
        # Preparar dados
        if not self.preparar_dados(df):
            return
        
        # Escolher tipo de modelo
        print("\n🤖 TIPO DE MODELO:")
        print("1. Random Forest (simples)")
        print("2. Ensemble (Random Forest + SVM + KNN)")
        
        try:
            tipo_escolha = input("\nEscolha o tipo (1 ou 2): ").strip()
            if tipo_escolha == '1':
                tipo = 'simples'
                self.treinar_modelo_simples()
            elif tipo_escolha == '2':
                tipo = 'ensemble'
                self.treinar_modelo_ensemble()
            else:
                print("❌ Escolha inválida")
                return
        except:
            print("❌ Entrada inválida")
            return
        
        # Validação cruzada
        cv_score = self.validacao_cruzada()
        
        # Avaliação detalhada
        self.avaliar_modelo()
        
        # Salvar modelo
        modelo_file, scaler_file, info_file = self.salvar_modelo(tipo)
        
        print(f"\n🎉 TREINAMENTO CONCLUÍDO!")
        print(f"📊 Acurácia final: {self.accuracy:.3f}")
        print(f"📊 Validação cruzada: {cv_score:.3f}")
        print(f"💾 Modelo salvo em: {modelo_file}")

if __name__ == "__main__":
    treinador = TreinadorModeloLIBRAS()
    treinador.executar()

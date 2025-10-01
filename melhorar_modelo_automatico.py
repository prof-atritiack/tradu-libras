#!/usr/bin/env python3
"""
Melhorador Automático de Modelo para TraduLibras
Usa técnicas avançadas para melhorar precisão sem coleta manual extensiva
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime
import random

class MelhoradorAutomatico:
    def __init__(self):
        self.modelo_atual = None
        self.dados_originais = []
        self.dados_aumentados = []
        
    def carregar_modelo_atual(self):
        """Carrega o modelo atual"""
        try:
            with open('modelos/modelo_aprimorado_20251001_115726.pkl', 'rb') as f:
                self.modelo_atual = pickle.load(f)
            print("✅ Modelo atual carregado")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def gerar_dados_sinteticos(self, features_base, letra, num_variacoes=100):
        """Gera dados sintéticos com variações realistas"""
        variacoes = []
        
        for _ in range(num_variacoes):
            features_variadas = features_base.copy()
            
            # Aplicar diferentes tipos de variação
            for i in range(len(features_variadas)):
                if i < 42:  # Coordenadas x,y principais
                    # Variação baseada na importância da coordenada
                    if i % 2 == 0:  # Coordenada X
                        variacao = random.gauss(0, 0.02)  # Variação menor para X
                    else:  # Coordenada Y
                        variacao = random.gauss(0, 0.03)  # Variação maior para Y
                else:  # Distâncias
                    variacao = random.gauss(0, 0.01)  # Variação mínima para distâncias
                
                features_variadas[i] += variacao
            
            variacoes.append(features_variadas)
        
        return variacoes
    
    def criar_dados_balanceados(self):
        """Cria dados balanceados para todas as letras"""
        letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
        
        # Dados sintéticos baseados em padrões conhecidos
        dados_sinteticos = []
        
        for letra in letras:
            # Gerar dados base sintéticos para cada letra
            features_base = self.gerar_features_base_letra(letra)
            
            # Gerar variações
            variacoes = self.gerar_dados_sinteticos(features_base, letra, 200)
            
            for features in variacoes:
                dados_sinteticos.append({
                    'features': features,
                    'letra': letra
                })
        
        print(f"✅ Gerados {len(dados_sinteticos)} dados sintéticos balanceados")
        return dados_sinteticos
    
    def gerar_features_base_letra(self, letra):
        """Gera features base sintéticas para cada letra baseado em padrões conhecidos"""
        # Features base (51 features)
        features = [0.0] * 51
        
        # Padrões específicos para cada letra (baseados em conhecimento de LIBRAS)
        if letra == 'A':
            # A: polegar estendido, outros dedos fechados
            features[0] = 0.1   # thumb_x
            features[1] = -0.1 # thumb_y
            features[8] = -0.05  # index_x
            features[9] = 0.05   # index_y
            
        elif letra == 'B':
            # B: todos os dedos estendidos
            features[8] = 0.1   # index_x
            features[9] = -0.1  # index_y
            features[12] = 0.1  # middle_x
            features[13] = -0.1 # middle_y
            features[16] = 0.1  # ring_x
            features[17] = -0.1 # ring_y
            features[20] = 0.1  # pinky_x
            features[21] = -0.1 # pinky_y
            
        elif letra == 'C':
            # C: curvatura em C
            features[8] = 0.05   # index_x
            features[9] = -0.05  # index_y
            features[12] = 0.05  # middle_x
            features[13] = -0.05 # middle_y
            features[16] = 0.05  # ring_x
            features[17] = -0.05 # ring_y
            features[20] = 0.05  # pinky_x
            features[21] = -0.05 # pinky_y
            
        elif letra == 'D':
            # D: indicador estendido, outros fechados
            features[8] = 0.1   # index_x
            features[9] = -0.1  # index_y
            
        elif letra == 'E':
            # E: todos os dedos fechados
            features[0] = 0.05   # thumb_x
            features[1] = 0.05   # thumb_y
            
        elif letra == 'F':
            # F: indicador e polegar se tocam
            features[0] = 0.1   # thumb_x
            features[1] = -0.1  # thumb_y
            features[8] = 0.1   # index_x
            features[9] = -0.1  # index_y
            
        elif letra == 'G':
            # G: indicador apontando
            features[8] = 0.1   # index_x
            features[9] = -0.1  # index_y
            
        elif letra == 'I':
            # I: mindinho estendido
            features[20] = 0.1  # pinky_x
            features[21] = -0.1 # pinky_y
            
        elif letra == 'L':
            # L: indicador e polegar em L
            features[0] = 0.1   # thumb_x
            features[1] = -0.1  # thumb_y
            features[8] = 0.1   # index_x
            features[9] = -0.1  # index_y
            
        elif letra == 'M':
            # M: três dedos estendidos
            features[12] = 0.1  # middle_x
            features[13] = -0.1 # middle_y
            features[16] = 0.1  # ring_x
            features[17] = -0.1 # ring_y
            features[20] = 0.1  # pinky_x
            features[21] = -0.1 # pinky_y
            
        elif letra == 'N':
            # N: dois dedos estendidos
            features[16] = 0.1  # ring_x
            features[17] = -0.1 # ring_y
            features[20] = 0.1  # pinky_x
            features[21] = -0.1 # pinky_y
            
        elif letra == 'O':
            # O: todos os dedos curvados em O
            features[8] = 0.05   # index_x
            features[9] = -0.05  # index_y
            features[12] = 0.05  # middle_x
            features[13] = -0.05 # middle_y
            features[16] = 0.05  # ring_x
            features[17] = -0.05 # ring_y
            features[20] = 0.05  # pinky_x
            features[21] = -0.05 # pinky_y
            
        elif letra == 'P':
            # P: indicador e polegar em P
            features[0] = 0.1   # thumb_x
            features[1] = -0.1  # thumb_y
            features[8] = 0.1   # index_x
            features[9] = -0.1  # index_y
            
        elif letra == 'Q':
            # Q: indicador e polegar em Q
            features[0] = 0.1   # thumb_x
            features[1] = -0.1  # thumb_y
            features[8] = 0.1   # index_x
            features[9] = -0.1  # index_y
            
        elif letra == 'R':
            # R: indicador e médio cruzados
            features[8] = 0.1   # index_x
            features[9] = -0.1  # index_y
            features[12] = 0.1  # middle_x
            features[13] = -0.1 # middle_y
            
        elif letra == 'S':
            # S: punho fechado
            features[0] = 0.05   # thumb_x
            features[1] = 0.05   # thumb_y
            
        elif letra == 'T':
            # T: polegar entre indicador e médio
            features[0] = 0.1   # thumb_x
            features[1] = -0.1  # thumb_y
            features[8] = 0.1   # index_x
            features[9] = -0.1  # index_y
            features[12] = 0.1  # middle_x
            features[13] = -0.1 # middle_y
            
        elif letra == 'U':
            # U: indicador e médio estendidos
            features[8] = 0.1   # index_x
            features[9] = -0.1  # index_y
            features[12] = 0.1  # middle_x
            features[13] = -0.1 # middle_y
            
        elif letra == 'V':
            # V: indicador e médio em V
            features[8] = 0.1   # index_x
            features[9] = -0.1  # index_y
            features[12] = 0.1  # middle_x
            features[13] = -0.1 # middle_y
            
        elif letra == 'W':
            # W: três dedos estendidos
            features[8] = 0.1   # index_x
            features[9] = -0.1  # index_y
            features[12] = 0.1  # middle_x
            features[13] = -0.1 # middle_y
            features[16] = 0.1  # ring_x
            features[17] = -0.1 # ring_y
            
        elif letra == 'Y':
            # Y: mindinho e polegar estendidos
            features[0] = 0.1   # thumb_x
            features[1] = -0.1  # thumb_y
            features[20] = 0.1  # pinky_x
            features[21] = -0.1 # pinky_y
        
        # Adicionar features de distância baseadas no padrão
        # Distâncias entre dedos e pulso
        features[42] = abs(features[0]) + abs(features[1])   # thumb
        features[43] = abs(features[8]) + abs(features[9])  # index
        features[44] = abs(features[12]) + abs(features[13]) # middle
        features[45] = abs(features[16]) + abs(features[17]) # ring
        features[46] = abs(features[20]) + abs(features[21]) # pinky
        
        # Distâncias entre dedos
        features[47] = abs(features[0] - features[8]) + abs(features[1] - features[9])   # thumb-index
        features[48] = abs(features[8] - features[12]) + abs(features[9] - features[13]) # index-middle
        features[49] = abs(features[12] - features[16]) + abs(features[13] - features[17]) # middle-ring
        features[50] = abs(features[16] - features[20]) + abs(features[17] - features[21]) # ring-pinky
        
        return features
    
    def treinar_modelo_ensemble(self, dados):
        """Treina modelo ensemble com múltiplos algoritmos"""
        print("🤖 Treinando modelo ensemble...")
        
        # Preparar dados
        X = [d['features'] for d in dados]
        y = [d['letra'] for d in dados]
        
        # Normalizar features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Criar modelos individuais
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42
        )
        
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        
        knn = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean'
        )
        
        # Criar ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('svm', svm),
                ('knn', knn)
            ],
            voting='soft'  # Usar probabilidades
        )
        
        # Treinar ensemble
        ensemble.fit(X_scaled, y)
        
        # Avaliar com validação cruzada
        scores = cross_val_score(ensemble, X_scaled, y, cv=5, scoring='accuracy')
        print(f"✅ Precisão média (CV): {scores.mean():.2%} (+/- {scores.std() * 2:.2%})")
        
        return ensemble, scaler
    
    def salvar_modelo_melhorado(self, modelo, scaler):
        """Salva o modelo melhorado"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Criar diretório se não existir
        os.makedirs('modelos', exist_ok=True)
        
        # Salvar modelo
        model_path = f'modelos/modelo_ensemble_{timestamp}.pkl'
        scaler_path = f'modelos/scaler_ensemble_{timestamp}.pkl'
        info_path = f'modelos/modelo_info_ensemble_{timestamp}.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(modelo, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Informações do modelo
        model_info = {
            'classes': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'],
            'features_count': 51,
            'model_type': 'Ensemble_VotingClassifier',
            'algorithms': ['RandomForest', 'SVM', 'KNN'],
            'training_date': timestamp,
            'requires_scaling': True
        }
        
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"💾 Modelo salvo: {model_path}")
        print(f"💾 Scaler salvo: {scaler_path}")
        print(f"💾 Info salva: {info_path}")
        
        return model_path, scaler_path, info_path
    
    def executar_melhoria(self):
        """Executa processo completo de melhoria"""
        print("🚀 Melhorador Automático de Modelo - TraduLibras")
        print("=" * 60)
        
        # Carregar modelo atual
        if not self.carregar_modelo_atual():
            print("⚠️ Continuando sem modelo atual...")
        
        # Gerar dados sintéticos balanceados
        print("\n📊 Gerando dados sintéticos balanceados...")
        dados_sinteticos = self.criar_dados_balanceados()
        
        # Treinar modelo ensemble
        print("\n🤖 Treinando modelo ensemble...")
        modelo_ensemble, scaler = self.treinar_modelo_ensemble(dados_sinteticos)
        
        # Salvar modelo melhorado
        print("\n💾 Salvando modelo melhorado...")
        model_path, scaler_path, info_path = self.salvar_modelo_melhorado(modelo_ensemble, scaler)
        
        print("\n🎉 Melhoria concluída com sucesso!")
        print("\n📋 Próximos passos:")
        print("1. Atualize o app.py para usar o novo modelo ensemble")
        print("2. Adicione o scaler para normalização")
        print("3. Teste a precisão melhorada")
        print("4. Ajuste parâmetros de estabilização se necessário")
        
        return True

def main():
    """Função principal"""
    melhorador = MelhoradorAutomatico()
    melhorador.executar_melhoria()

if __name__ == "__main__":
    main()

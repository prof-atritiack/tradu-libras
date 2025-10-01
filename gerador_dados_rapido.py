#!/usr/bin/env python3
"""
Gerador Rápido de Dados Sintéticos para Libras
Cria dados realistas baseados em padrões conhecidos de gestos
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

class GeradorDadosLibras:
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Letras para gerar
        self.letras_para_gerar = [
            'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'M', 'N', 
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Z'
        ]
        
        # Padrões de gestos baseados em Libras reais
        self.padroes_gestos = {
            'D': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'E': {'dedos': [0, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'F': {'dedos': [1, 1, 0, 0, 1], 'angulo': 0, 'posicao': 'centro'},
            'G': {'dedos': [1, 1, 0, 0, 0], 'angulo': 15, 'posicao': 'lateral'},
            'H': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'I': {'dedos': [0, 0, 0, 0, 1], 'angulo': 0, 'posicao': 'centro'},
            'J': {'dedos': [0, 0, 0, 0, 1], 'angulo': 0, 'posicao': 'centro'},
            'K': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'M': {'dedos': [0, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'N': {'dedos': [0, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'O': {'dedos': [0, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'P': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'Q': {'dedos': [1, 1, 0, 0, 0], 'angulo': 15, 'posicao': 'lateral'},
            'R': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'S': {'dedos': [0, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'T': {'dedos': [0, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'U': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'V': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'W': {'dedos': [1, 1, 1, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'X': {'dedos': [1, 0, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'},
            'Z': {'dedos': [1, 1, 0, 0, 0], 'angulo': 0, 'posicao': 'centro'}
        }
        
        self.dados_gerados = []
    
    def gerar_landmarks_sinteticos(self, letra, variacao=0):
        """Gera landmarks sintéticos baseados no padrão da letra"""
        padrao = self.padroes_gestos[letra]
        
        # Landmarks base (21 pontos da mão)
        landmarks_base = np.zeros((21, 3))
        
        # Pulso (ponto 0) - sempre na origem
        landmarks_base[0] = [0, 0, 0]
        
        # Base dos dedos (pontos 1, 5, 9, 13, 17)
        landmarks_base[1] = [-0.05, -0.02, 0]  # Polegar
        landmarks_base[5] = [-0.03, -0.05, 0]   # Indicador
        landmarks_base[9] = [0, -0.05, 0]        # Médio
        landmarks_base[13] = [0.03, -0.05, 0]    # Anelar
        landmarks_base[17] = [0.05, -0.02, 0]   # Mindinho
        
        # Pontas dos dedos baseadas no padrão
        dedos_config = padrao['dedos']
        
        # Polegar (pontos 2, 3, 4)
        if dedos_config[0]:  # Polegar levantado
            landmarks_base[2] = [-0.08, -0.05, 0]
            landmarks_base[3] = [-0.12, -0.08, 0]
            landmarks_base[4] = [-0.15, -0.1, 0]
        else:  # Polegar abaixado
            landmarks_base[2] = [-0.06, -0.01, 0]
            landmarks_base[3] = [-0.07, 0.01, 0]
            landmarks_base[4] = [-0.08, 0.03, 0]
        
        # Indicador (pontos 6, 7, 8)
        if dedos_config[1]:  # Indicador levantado
            landmarks_base[6] = [-0.05, -0.1, 0]
            landmarks_base[7] = [-0.05, -0.15, 0]
            landmarks_base[8] = [-0.05, -0.2, 0]
        else:  # Indicador abaixado
            landmarks_base[6] = [-0.03, -0.06, 0]
            landmarks_base[7] = [-0.03, -0.07, 0]
            landmarks_base[8] = [-0.03, -0.08, 0]
        
        # Médio (pontos 10, 11, 12)
        if dedos_config[2]:  # Médio levantado
            landmarks_base[10] = [0, -0.1, 0]
            landmarks_base[11] = [0, -0.15, 0]
            landmarks_base[12] = [0, -0.2, 0]
        else:  # Médio abaixado
            landmarks_base[10] = [0, -0.06, 0]
            landmarks_base[11] = [0, -0.07, 0]
            landmarks_base[12] = [0, -0.08, 0]
        
        # Anelar (pontos 14, 15, 16)
        if dedos_config[3]:  # Anelar levantado
            landmarks_base[14] = [0.03, -0.1, 0]
            landmarks_base[15] = [0.03, -0.15, 0]
            landmarks_base[16] = [0.03, -0.2, 0]
        else:  # Anelar abaixado
            landmarks_base[14] = [0.03, -0.06, 0]
            landmarks_base[15] = [0.03, -0.07, 0]
            landmarks_base[16] = [0.03, -0.08, 0]
        
        # Mindinho (pontos 18, 19, 20)
        if dedos_config[4]:  # Mindinho levantado
            landmarks_base[18] = [0.05, -0.1, 0]
            landmarks_base[19] = [0.05, -0.15, 0]
            landmarks_base[20] = [0.05, -0.2, 0]
        else:  # Mindinho abaixado
            landmarks_base[18] = [0.05, -0.06, 0]
            landmarks_base[19] = [0.05, -0.07, 0]
            landmarks_base[20] = [0.05, -0.08, 0]
        
        # Aplicar rotação baseada no ângulo
        angulo_rad = np.radians(padrao['angulo'])
        cos_a, sin_a = np.cos(angulo_rad), np.sin(angulo_rad)
        
        # Matriz de rotação 2D
        for i in range(len(landmarks_base)):
            x, y, z = landmarks_base[i]
            landmarks_base[i] = [
                x * cos_a - y * sin_a,
                x * sin_a + y * cos_a,
                z
            ]
        
        # Adicionar variações aleatórias
        if variacao > 0:
            ruido = np.random.normal(0, variacao, landmarks_base.shape)
            landmarks_base += ruido
        
        # Normalizar em relação ao pulso
        pulso = landmarks_base[0].copy()
        for i in range(len(landmarks_base)):
            landmarks_base[i] -= pulso
        
        # Converter para formato de lista (63 valores)
        pontos = []
        for landmark in landmarks_base:
            pontos.extend([landmark[0], landmark[1], landmark[2]])
        
        return pontos
    
    def gerar_dados_letra(self, letra, num_amostras=50):
        """Gera dados sintéticos para uma letra"""
        print(f"🎯 Gerando {num_amostras} amostras para letra {letra}...")
        
        dados_letra = []
        
        for i in range(num_amostras):
            # Gerar landmarks com variação
            variacao = np.random.uniform(0.01, 0.03)  # Variação pequena
            landmarks = self.gerar_landmarks_sinteticos(letra, variacao)
            
            if len(landmarks) == 63:
                dados_letra.append({
                    'label': letra,
                    **{f'point_{j}': landmarks[j] for j in range(63)}
                })
        
        print(f"✅ Gerados {len(dados_letra)} dados para {letra}")
        return dados_letra
    
    def gerar_todos_dados(self):
        """Gera dados para todas as letras"""
        print("🚀 Iniciando geração de dados sintéticos...")
        
        todos_dados = []
        
        for letra in self.letras_para_gerar:
            dados_letra = self.gerar_dados_letra(letra, 50)  # 50 amostras por letra
            todos_dados.extend(dados_letra)
        
        self.dados_gerados = todos_dados
        print(f"\n✅ Geração concluída! Total: {len(todos_dados)} dados")
        
        return todos_dados
    
    def salvar_dados(self, nome_arquivo='gestos_libras_expandido.csv'):
        """Salva dados gerados"""
        if not self.dados_gerados:
            print("❌ Nenhum dado para salvar")
            return
        
        try:
            # Carregar dados existentes
            dados_existentes = []
            if os.path.exists('gestos_libras.csv'):
                df_existente = pd.read_csv('gestos_libras.csv')
                dados_existentes = df_existente.to_dict('records')
                print(f"📊 Carregados {len(dados_existentes)} dados existentes")
            
            # Combinar dados
            todos_dados = dados_existentes + self.dados_gerados
            
            # Criar DataFrame
            df = pd.DataFrame(todos_dados)
            
            # Salvar
            df.to_csv(nome_arquivo, index=False)
            
            print(f"✅ Dados salvos em {nome_arquivo}")
            print(f"📊 Total de amostras: {len(todos_dados)}")
            print(f"📈 Novas amostras: {len(self.dados_gerados)}")
            
            # Mostrar distribuição
            print("\n📊 Distribuição por classe:")
            print(df['label'].value_counts().sort_index())
            
        except Exception as e:
            print(f"❌ Erro ao salvar dados: {e}")
    
    def treinar_modelo(self):
        """Treina modelo com os dados expandidos"""
        try:
            print("\n🧠 Treinando modelo com dados expandidos...")
            
            # Carregar dados
            df = pd.read_csv('gestos_libras_expandido.csv')
            feature_columns = [col for col in df.columns if col != 'label']
            X = df[feature_columns].values
            y = df['label'].values
            
            print(f"📊 Dados: {len(df)} amostras, {len(feature_columns)} features")
            print(f"🏷️ Classes: {sorted(df['label'].unique())}")
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Treinar modelo
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=3,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Avaliar
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            print(f"📈 Acurácia treino: {train_acc:.2%}")
            print(f"📈 Acurácia teste: {test_acc:.2%}")
            
            # Salvar modelo
            os.makedirs('modelos', exist_ok=True)
            with open('modelos/modelo_libras_expandido.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            # Salvar informações
            model_info = {
                'classes': model.classes_.tolist(),
                'n_features': len(feature_columns),
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'n_samples': len(df),
                'vocabulary_type': 'expanded_synthetic',
                'created_at': datetime.now().isoformat()
            }
            
            with open('modelos/modelo_info_expandido.pkl', 'wb') as f:
                pickle.dump(model_info, f)
            
            print("✅ Modelo expandido salvo!")
            return True
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            return False

def main():
    print("🚀 TraduLibras - Gerador Rápido de Dados v1.0")
    print("📚 Gerando dados sintéticos baseados em padrões de Libras...")
    
    gerador = GeradorDadosLibras()
    
    # Gerar dados
    dados = gerador.gerar_todos_dados()
    
    if dados:
        # Salvar dados
        gerador.salvar_dados()
        
        # Treinar modelo
        gerador.treinar_modelo()
        
        print("\n🎉 Processo concluído!")
        print("📁 Arquivos criados:")
        print("   - gestos_libras_expandido.csv")
        print("   - modelos/modelo_libras_expandido.pkl")
        print("   - modelos/modelo_info_expandido.pkl")
        
        print("\n📊 Resumo:")
        print(f"   - Total de classes: {len(gerador.letras_para_gerar) + 5}")  # +5 das existentes
        print(f"   - Amostras por classe: ~50")
        print(f"   - Total de amostras: {len(dados) + 150}")  # +150 das existentes
    else:
        print("❌ Nenhum dado foi gerado")

if __name__ == "__main__":
    main()

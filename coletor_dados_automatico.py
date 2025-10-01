#!/usr/bin/env python3
"""
Coletor Automático de Dados para TraduLibras
Gera dados sintéticos e variações para melhorar o modelo sem coleta manual
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class ColetorAutomatico:
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Letras suportadas
        self.letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
        
        # Dados coletados
        self.dados = []
        
    def process_landmarks(self, hand_landmarks):
        """Processa landmarks com features avançadas"""
        if not hand_landmarks:
            return None

        # Ponto de referência (pulso)
        wrist = hand_landmarks.landmark[0]

        # Extrair coordenadas x,y relativas ao pulso (42 features)
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y
            ])

        # Adicionar features extras para melhorar precisão
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]

        # Distâncias entre dedos e pulso
        features.extend([
            abs(thumb_tip.x - wrist.x) + abs(thumb_tip.y - wrist.y),
            abs(index_tip.x - wrist.x) + abs(index_tip.y - wrist.y),
            abs(middle_tip.x - wrist.x) + abs(middle_tip.y - wrist.y),
            abs(ring_tip.x - wrist.x) + abs(ring_tip.y - wrist.y),
            abs(pinky_tip.x - wrist.x) + abs(pinky_tip.y - wrist.y)
        ])

        # Distâncias entre dedos
        features.extend([
            abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y),
            abs(index_tip.x - middle_tip.x) + abs(index_tip.y - middle_tip.y),
            abs(middle_tip.x - ring_tip.x) + abs(middle_tip.y - ring_tip.y),
            abs(ring_tip.x - pinky_tip.x) + abs(ring_tip.y - pinky_tip.y)
        ])

        return features  # Total: 51 features

    def gerar_variacoes_sinteticas(self, features_base, letra, num_variacoes=50):
        """Gera variações sintéticas dos dados para aumentar robustez"""
        variacoes = []
        
        for _ in range(num_variacoes):
            # Criar cópia das features base
            features_variadas = features_base.copy()
            
            # Aplicar pequenas variações aleatórias
            for i in range(len(features_variadas)):
                # Variação de ±5% para coordenadas principais
                if i < 42:  # Coordenadas x,y
                    variacao = random.uniform(-0.05, 0.05)
                else:  # Distâncias
                    variacao = random.uniform(-0.02, 0.02)
                
                features_variadas[i] += variacao
            
            variacoes.append({
                'features': features_variadas,
                'letra': letra
            })
        
        return variacoes

    def coletar_dados_existentes(self):
        """Carrega dados existentes se disponíveis"""
        try:
            # Tentar carregar dados existentes
            if os.path.exists('gestos_libras_aprimorado.csv'):
                df = pd.read_csv('gestos_libras_aprimorado.csv')
                print(f"✅ Carregados {len(df)} dados existentes")
                
                for _, row in df.iterrows():
                    features = [row[f'feature_{i}'] for i in range(51)]
                    self.dados.append({
                        'features': features,
                        'letra': row['letra']
                    })
                return True
        except Exception as e:
            print(f"⚠️ Erro ao carregar dados existentes: {e}")
        
        return False

    def coletar_dados_camera(self, letras_especificas=None, amostras_por_letra=100):
        """Coleta dados da câmera com interface simplificada"""
        if letras_especificas is None:
            letras_especificas = ['A', 'E', 'C', 'D']  # Letras mais problemáticas
        
        print("🎥 Iniciando coleta automática de dados...")
        print("📋 Letras a coletar:", letras_especificas)
        print("📊 Amostras por letra:", amostras_por_letra)
        
        # Inicializar câmera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Erro: Não foi possível abrir a câmera")
            return False
        
        for letra in letras_especificas:
            print(f"\n📝 Coletando dados para letra: {letra}")
            print("   Posicione sua mão e pressione ESPAÇO para capturar")
            print("   Pressione 'q' para pular esta letra")
            
            amostras_coletadas = 0
            
            while amostras_coletadas < amostras_por_letra:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Flip horizontal para espelho
                frame = cv2.flip(frame, 1)
                
                # Processar frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Desenhar landmarks se detectados
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                
                # Mostrar instruções
                cv2.putText(frame, f"Letra: {letra}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Amostras: {amostras_coletadas}/{amostras_por_letra}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "ESPACO: Capturar | Q: Pular", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('Coleta Automática - TraduLibras', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Espaço para capturar
                    if results.multi_hand_landmarks:
                        landmarks = results.multi_hand_landmarks[0]
                        features = self.process_landmarks(landmarks)
                        
                        if features and len(features) == 51:
                            # Adicionar dados base
                            self.dados.append({
                                'features': features,
                                'letra': letra
                            })
                            
                            # Gerar variações sintéticas
                            variacoes = self.gerar_variacoes_sinteticas(features, letra, 10)
                            self.dados.extend(variacoes)
                            
                            amostras_coletadas += 1
                            print(f"   ✅ Amostra {amostras_coletadas} coletada + 10 variações")
                        else:
                            print("   ⚠️ Landmarks inválidos, tente novamente")
                    else:
                        print("   ⚠️ Mão não detectada, tente novamente")
                
                elif key == ord('q'):  # Q para pular
                    print(f"   ⏭️ Pulando letra {letra}")
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        return True

    def treinar_modelo_melhorado(self):
        """Treina modelo com dados coletados e variações"""
        if not self.dados:
            print("❌ Nenhum dado coletado!")
            return False
        
        print(f"\n🤖 Treinando modelo com {len(self.dados)} amostras...")
        
        # Preparar dados
        X = [d['features'] for d in self.dados]
        y = [d['letra'] for d in self.dados]
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Treinar modelo com parâmetros otimizados
        model = RandomForestClassifier(
            n_estimators=300,  # Mais árvores
            max_depth=20,      # Profundidade maior
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Balancear classes
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Avaliar modelo
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Modelo treinado com precisão: {accuracy:.2%}")
        print("\n📊 Relatório detalhado:")
        print(classification_report(y_test, y_pred))
        
        # Salvar modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'modelos/modelo_automatico_{timestamp}.pkl'
        info_path = f'modelos/modelo_info_automatico_{timestamp}.pkl'
        
        # Criar diretório se não existir
        os.makedirs('modelos', exist_ok=True)
        
        # Salvar modelo
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Salvar informações
        model_info = {
            'classes': self.letras,
            'features_count': 51,
            'test_accuracy': accuracy,
            'total_samples': len(self.dados),
            'training_date': timestamp,
            'model_type': 'RandomForestClassifier_Automatico'
        }
        
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"💾 Modelo salvo: {model_path}")
        print(f"💾 Info salva: {info_path}")
        
        return True

    def executar_coleta_completa(self):
        """Executa processo completo de coleta e treinamento"""
        print("🚀 Coletor Automático de Dados - TraduLibras")
        print("=" * 50)
        
        # Carregar dados existentes
        dados_existentes = self.coletar_dados_existentes()
        
        # Coletar novos dados
        print("\n📝 Iniciando coleta de dados...")
        sucesso = self.coletar_dados_camera()
        
        if not sucesso:
            print("❌ Falha na coleta de dados")
            return False
        
        # Treinar modelo
        print("\n🤖 Treinando modelo melhorado...")
        sucesso_treinamento = self.treinar_modelo_melhorado()
        
        if sucesso_treinamento:
            print("\n🎉 Processo concluído com sucesso!")
            print("📋 Próximos passos:")
            print("1. Atualize o app.py para usar o novo modelo")
            print("2. Teste a precisão melhorada")
            print("3. Ajuste parâmetros se necessário")
        else:
            print("❌ Falha no treinamento do modelo")
        
        return sucesso_treinamento

def main():
    """Função principal"""
    coletor = ColetorAutomatico()
    coletor.executar_coleta_completa()

if __name__ == "__main__":
    main()

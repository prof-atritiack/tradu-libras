#!/usr/bin/env python3
"""
Coletor de Dados para Treinamento LIBRAS
Este script permite coletar dados de gestos LIBRAS para treinar novos modelos
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import pickle
from datetime import datetime
import time

class ColetorLIBRAS:
    def __init__(self, pasta_dados='dados_coletados'):
        """Inicializar coletor"""
        self.pasta_dados = pasta_dados
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Criar pasta de dados se não existir
        if not os.path.exists(pasta_dados):
            os.makedirs(pasta_dados)
            
        # Lista para armazenar dados
        self.dados_coletados = []
        self.classe_atual = None
        self.contador_amostras = 0
        
    def processar_landmarks(self, hand_landmarks):
        """Processar landmarks da mão"""
        if not hand_landmarks:
            return None
            
        # Ponto de referência (pulso)
        wrist = hand_landmarks.landmark[0]
        
        # Features básicas
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y
            ])
        
        # Features extras
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
            abs(ring_tip.x - pinky_tip.x) + abs(pinky_tip.y - pinky_tip.y)
        ])
        
        return features  # Total: 51 features
    
    def mostrar_status(self, frame, classe, contador):
        """Mostrar status na imagem"""
        cv2.putText(frame, f"CLASSE: {classe}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"AMOSTRAS: {contador}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"TOTAL: {len(self.dados_coletados)}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Próxima classe | ESC: Sair", (10, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if classe:
            cv2.putText(frame, "Detectando gesto...", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Pressione SPACE para definir classe", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    def coletar_dados(self):
        """Função principal de coleta"""
        print("=" * 60)
        print("COLETOR DE DADOS LIBRAS")
        print("=" * 60)
        print()
        print("Instruções:")
        print("1. Posicione sua mão na câmera")
        print("2. Pressione SPACE para definir a letra atual")
        print("3. Faça o gesto e aguarde detecção automática")
        print("4. Pressione SPACE para próxima letra")
        print("5. ESC para sair e salvar dados")
        print()
        
        # Inicializar câmera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ ERRO: Não foi possível acessar a câmera!")
            return
        
        print("✅ Câmera inicializada!")
        print("Aguardando definição da primeira classe...")
        print()
        
        cooldown_detecacao = 0.5  # 0.5 segundos entre detecções
        ultima_detecacao = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip horizontal
            frame = cv2.flip(frame, 1)
            
            # Converter para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processar com MediaPipe
            results = self.hands.process(rgb_frame)
            
            current_time = time.time()
            
            # Desenhar landmarks se detectado
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                               self.mp_hands.HAND_CONNECTIONS)
                
                # Processar landmarks se classe definida e cooldown ok
                if self.classe_atual and (current_time - ultima_detecacao) >= cooldown_detecacao:
                    points = self.processar_landmarks(hand_landmarks)
                    
                    if points and len(points) == 51:
                        # Adicionar dados
                        data_row = [self.classe_atual] + points
                        self.dados_coletados.append(data_row)
                        self.contador_amostras += 1
                        ultima_detecacao = current_time
                        print(f"📝 Coletada amostra {self.contador_amostras} para '{self.classe_atual}'")
            
            # Mostrar status
            self.mostrar_status(frame, self.classe_atual, self.contador_amostras)
            
            # Mostrar frame
            cv2.imshow('Coletor LIBRAS', frame)
            
            # Processar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC - Sair
                break
            elif key == 32:  # SPACE - Próxima classe
                classe_anterior = self.classe_atual
                self.classe_atual = input(f"\nDigite a letra atual (anterior: {classe_anterior or 'nenhuma'}): ").strip().upper()
                
                if self.classe_atual:
                    self.contador_amostras = 0
                    print(f"🎯 Coletando para classe: {self.classe_atual}")
                else:
                    print("⚠️ Classe não definida")
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        
        # Salvar dados
        self.salvar_dados()
    
    def salvar_dados(self):
        """Salvar dados coletados"""
        if not self.dados_coletados:
            print("❌ Nenhum dado coletado!")
            return
        
        print("💾 Salvando dados...")
        
        # Criar DataFrame
        columns = ['gesture_type'] + [f'x{i}' for i in range(1, 43)] + [f'y{i}' for i in range(1, 43)] + \
                  [f'd{i}' for i in range(1, 10)]  # 51 features total
        
        df = pd.DataFrame(self.dados_coletados, columns=columns)
        
        # Nome do arquivo com timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'gestos_libras_coletados_{timestamp}.csv'
        filepath = os.path.join(self.pasta_dados, filename)
        
        # Salvar CSV
        df.to_csv(filepath, index=False)
        
        # Estatísticas
        total_amostras = len(df)
        classes_unicas = df['gesture_type'].unique()
        
        print()
        print("=" * 60)
        print("DADOS SALVOS COM SUCESSO!")
        print("=" * 60)
        print(f"📁 Arquivo: {filepath}")
        print(f"📊 Total de amostras: {total_amostras}")
        print(f"🎯 Classes coletadas: {', '.join(sorted(classes_unicas))}")
        print(f"📈 Média por classe: {total_amostras / len(classes_unicas):.1f}")
        print("=" * 60)

def main():
    """Função principal"""
    print("🚀 INICIANDO COLETOR DE DADOS LIBRAS")
    
    try:
        coletor = ColetorLIBRAS()
        coletor.coletar_dados()
    except KeyboardInterrupt:
        print("\n⏹️ Coleta interrompida pelo usuário")
    except Exception as e:
        print(f"❌ ERRO: {e}")
    
    print("👋 Programa finalizado")

if __name__ == "__main__":
    main()

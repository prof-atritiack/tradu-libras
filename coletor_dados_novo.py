#!/usr/bin/env python3
"""
Coletor de Dados LIBRAS - Sistema Novo e Melhorado
Coleta dados de gestos LIBRAS com interface amigável e validação
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import os
import json

class ColetorDadosLIBRAS:
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Lista de gestos LIBRAS
        self.gestos = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'ESPACO'
        ]
        
        # Dados coletados
        self.dados_coletados = []
        self.gesto_atual = None
        self.amostras_coletadas = 0
        self.amostras_necessarias = 50  # Reduzido para coleta mais rápida
        
        # Interface
        self.camera = None
        self.coletando = False
        self.pausado = False
        
        print("🎯 Coletor de Dados LIBRAS - Sistema Novo")
        print("=" * 50)
        
    def inicializar_camera(self):
        """Inicializar câmera com diferentes índices"""
        for camera_index in [0, 1, 2]:
            try:
                self.camera = cv2.VideoCapture(camera_index)
                if self.camera.isOpened():
                    ret, test_frame = self.camera.read()
                    if ret and test_frame is not None:
                        print(f"✅ Câmera inicializada com índice {camera_index}")
                        return True
                    else:
                        self.camera.release()
                        self.camera = None
            except Exception as e:
                print(f"❌ Erro ao inicializar câmera {camera_index}: {e}")
                if self.camera:
                    self.camera.release()
                    self.camera = None
        
        print("❌ Nenhuma câmera disponível!")
        return False
    
    def processar_landmarks(self, hand_landmarks):
        """Processar landmarks da mão para features"""
        if not hand_landmarks:
            return None
        
        # Ponto de referência (pulso)
        wrist = hand_landmarks.landmark[0]
        
        # Features básicas: coordenadas x,y relativas ao pulso
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y
            ])
        
        # Features extras para melhor precisão
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
        
        # Distâncias entre dedos adjacentes
        features.extend([
            abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y),
            abs(index_tip.x - middle_tip.x) + abs(index_tip.y - middle_tip.y),
            abs(middle_tip.x - ring_tip.x) + abs(middle_tip.y - ring_tip.y),
            abs(ring_tip.x - pinky_tip.x) + abs(ring_tip.y - pinky_tip.y)
        ])
        
        return features  # Total: 42 + 5 + 4 = 51 features
    
    def mostrar_menu(self):
        """Mostrar menu de opções"""
        print("\n" + "=" * 50)
        print("📋 MENU DE COLETA DE DADOS")
        print("=" * 50)
        print("1. Iniciar coleta de dados")
        print("2. Ver estatísticas")
        print("3. Salvar dados coletados")
        print("4. Carregar dados existentes")
        print("5. Limpar dados")
        print("6. Sair")
        print("=" * 50)
        
    def mostrar_gestos(self):
        """Mostrar lista de gestos disponíveis"""
        print("\n📝 GESTOS DISPONÍVEIS:")
        print("-" * 30)
        for i, gesto in enumerate(self.gestos):
            print(f"{i+1:2d}. {gesto}")
        print("-" * 30)
    
    def coletar_dados_gesto(self, gesto):
        """Coletar dados para um gesto específico"""
        self.gesto_atual = gesto
        self.amostras_coletadas = 0
        
        print(f"\n🎯 COLETANDO DADOS PARA: {gesto}")
        print("=" * 40)
        print("📋 INSTRUÇÕES:")
        print(f"• Faça o gesto da letra '{gesto}'")
        print(f"• Mantenha a mão estável")
        print(f"• Colete {self.amostras_necessarias} amostras")
        print("• Pressione ESPAÇO para coletar")
        print("• Pressione ESC para cancelar")
        print("=" * 40)
        
        self.coletando = True
        
        while self.coletando and self.amostras_coletadas < self.amostras_necessarias:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Flip horizontal para espelho
            frame = cv2.flip(frame, 1)
            
            # Converter para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processar com MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Desenhar landmarks se detectado
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Interface visual
            cv2.putText(frame, f"GESTO: {gesto}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"AMOSTRAS: {self.amostras_coletadas}/{self.amostras_necessarias}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "ESPACO: Coletar | ESC: Cancelar", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Barra de progresso
            progress = self.amostras_coletadas / self.amostras_necessarias
            cv2.rectangle(frame, (10, 130), (int(10 + 300 * progress), 150), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 130), (310, 150), (255, 255, 255), 2)
            
            cv2.imshow('Coletor LIBRAS', frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Espaço para coletar
                if results.multi_hand_landmarks:
                    features = self.processar_landmarks(results.multi_hand_landmarks[0])
                    if features and len(features) == 51:
                        self.dados_coletados.append({
                            'gesto': gesto,
                            'features': features,
                            'timestamp': datetime.now().isoformat()
                        })
                        self.amostras_coletadas += 1
                        print(f"✅ Amostra {self.amostras_coletadas}/{self.amostras_necessarias} coletada para {gesto}")
                    else:
                        print("❌ Erro ao processar landmarks")
                else:
                    print("❌ Mão não detectada")
            elif key == 27:  # ESC para cancelar
                self.coletando = False
                print("❌ Coleta cancelada")
        
        cv2.destroyAllWindows()
        
        if self.amostras_coletadas == self.amostras_necessarias:
            print(f"✅ Coleta completa para {gesto}: {self.amostras_coletadas} amostras")
        else:
            print(f"⚠️ Coleta incompleta para {gesto}: {self.amostras_coletadas} amostras")
    
    def coletar_todos_gestos(self):
        """Coletar dados para todos os gestos"""
        print("\n🚀 INICIANDO COLETA COMPLETA")
        print("=" * 50)
        
        gestos_para_coletar = []
        for i, gesto in enumerate(self.gestos):
            print(f"{i+1:2d}. {gesto}")
            gestos_para_coletar.append(gesto)
        
        print("\n📋 SELECIONE OS GESTOS PARA COLETAR:")
        print("• Digite números separados por vírgula (ex: 1,2,3)")
        print("• Digite 'todos' para coletar todos")
        print("• Digite 'cancelar' para sair")
        
        escolha = input("\nSua escolha: ").strip().lower()
        
        if escolha == 'cancelar':
            return
        
        if escolha == 'todos':
            gestos_selecionados = gestos_para_coletar
        else:
            try:
                indices = [int(x.strip()) - 1 for x in escolha.split(',')]
                gestos_selecionados = [gestos_para_coletar[i] for i in indices if 0 <= i < len(gestos_para_coletar)]
            except:
                print("❌ Entrada inválida")
                return
        
        print(f"\n🎯 Coletando dados para: {', '.join(gestos_selecionados)}")
        
        for gesto in gestos_selecionados:
            self.coletar_dados_gesto(gesto)
            
            # Perguntar se quer continuar
            if gesto != gestos_selecionados[-1]:
                continuar = input(f"\nContinuar para o próximo gesto? (s/n): ").strip().lower()
                if continuar != 's':
                    break
    
    def ver_estatisticas(self):
        """Mostrar estatísticas dos dados coletados"""
        if not self.dados_coletados:
            print("❌ Nenhum dado coletado ainda")
            return
        
        print("\n📊 ESTATÍSTICAS DOS DADOS COLETADOS")
        print("=" * 50)
        
        # Contar por gesto
        gestos_count = {}
        for dado in self.dados_coletados:
            gesto = dado['gesto']
            gestos_count[gesto] = gestos_count.get(gesto, 0) + 1
        
        total_amostras = len(self.dados_coletados)
        print(f"📈 Total de amostras: {total_amostras}")
        print(f"📝 Total de gestos: {len(gestos_count)}")
        print("\n📋 Amostras por gesto:")
        for gesto, count in sorted(gestos_count.items()):
            print(f"  {gesto}: {count} amostras")
    
    def salvar_dados(self):
        """Salvar dados coletados"""
        if not self.dados_coletados:
            print("❌ Nenhum dado para salvar")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dados_libras_novo_{timestamp}.csv"
        
        # Converter para DataFrame
        df_data = []
        for dado in self.dados_coletados:
            row = {'gesto': dado['gesto']}
            for i, feature in enumerate(dado['features']):
                row[f'feature_{i}'] = feature
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(filename, index=False)
        
        print(f"✅ Dados salvos em: {filename}")
        print(f"📊 {len(self.dados_coletados)} amostras salvas")
    
    def carregar_dados(self):
        """Carregar dados existentes"""
        # Procurar arquivos CSV
        csv_files = [f for f in os.listdir('.') if f.startswith('dados_libras_novo_') and f.endswith('.csv')]
        
        if not csv_files:
            print("❌ Nenhum arquivo de dados encontrado")
            return
        
        print("\n📁 ARQUIVOS DISPONÍVEIS:")
        for i, file in enumerate(csv_files):
            print(f"{i+1}. {file}")
        
        try:
            escolha = int(input("\nEscolha o arquivo (número): ")) - 1
            if 0 <= escolha < len(csv_files):
                filename = csv_files[escolha]
                
                # Carregar dados
                df = pd.read_csv(filename)
                
                # Converter de volta para formato interno
                self.dados_coletados = []
                for _, row in df.iterrows():
                    features = [row[f'feature_{i}'] for i in range(51)]
                    self.dados_coletados.append({
                        'gesto': row['gesto'],
                        'features': features,
                        'timestamp': datetime.now().isoformat()
                    })
                
                print(f"✅ Dados carregados de: {filename}")
                print(f"📊 {len(self.dados_coletados)} amostras carregadas")
            else:
                print("❌ Escolha inválida")
        except:
            print("❌ Entrada inválida")
    
    def limpar_dados(self):
        """Limpar todos os dados coletados"""
        if not self.dados_coletados:
            print("❌ Nenhum dado para limpar")
            return
        
        confirmar = input(f"\n⚠️ Tem certeza que quer limpar {len(self.dados_coletados)} amostras? (s/n): ").strip().lower()
        if confirmar == 's':
            self.dados_coletados = []
            print("✅ Dados limpos")
        else:
            print("❌ Operação cancelada")
    
    def executar(self):
        """Executar o coletor de dados"""
        if not self.inicializar_camera():
            return
        
        while True:
            self.mostrar_menu()
            escolha = input("\nEscolha uma opção: ").strip()
            
            if escolha == '1':
                self.coletar_todos_gestos()
            elif escolha == '2':
                self.ver_estatisticas()
            elif escolha == '3':
                self.salvar_dados()
            elif escolha == '4':
                self.carregar_dados()
            elif escolha == '5':
                self.limpar_dados()
            elif escolha == '6':
                break
            else:
                print("❌ Opção inválida")
        
        # Limpar recursos
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("\n👋 Coletor finalizado!")

if __name__ == "__main__":
    coletor = ColetorDadosLIBRAS()
    coletor.executar()

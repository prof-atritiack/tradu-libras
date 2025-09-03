#!/usr/bin/env python3
"""
Script para expandir o vocabulário do TraduLibras
Adiciona mais letras e números ao reconhecimento
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from datetime import datetime

class VocabularioExpansor:
    def __init__(self):
        # Inicializar MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Vocabulário expandido
        self.letras_completas = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        
        self.numeros = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        # Letras já implementadas
        self.letras_implementadas = ['A', 'B', 'C', 'L', 'Y']
        
        # Letras a serem adicionadas
        self.letras_para_adicionar = [letra for letra in self.letras_completas 
                                    if letra not in self.letras_implementadas]
        
        self.dados_existentes = []
        self.carregar_dados_existentes()
    
    def carregar_dados_existentes(self):
        """Carrega dados existentes do arquivo CSV"""
        try:
            if os.path.exists('gestos_libras.csv'):
                df = pd.read_csv('gestos_libras.csv')
                self.dados_existentes = df.to_dict('records')
                print(f"✅ Carregados {len(self.dados_existentes)} dados existentes")
            else:
                print("⚠️ Arquivo gestos_libras.csv não encontrado")
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
    
    def processar_landmarks(self, hand_landmarks):
        """Processa landmarks da mão e normaliza"""
        p0 = hand_landmarks.landmark[0]  # Ponto de referência (pulso)
        points = []
        for landmark in hand_landmarks.landmark:
            points.extend([
                landmark.x - p0.x,
                landmark.y - p0.y,
                landmark.z - p0.z
            ])
        return points
    
    def coletar_gestos(self, vocabulario, nome_arquivo):
        """Coleta gestos para um vocabulário específico"""
        print(f"\n🎯 Iniciando coleta para: {', '.join(vocabulario)}")
        print("📋 Instruções:")
        print("- Posicione sua mão no centro da câmera")
        print("- Faça o gesto da letra/número correspondente")
        print("- Pressione ESPAÇO para capturar")
        print("- Pressione ESC para pular")
        print("- Pressione Q para sair")
        
        camera = cv2.VideoCapture(0)
        dados_coletados = []
        
        for item in vocabulario:
            print(f"\n📝 Coletando gestos para: {item}")
            contador = 0
            meta_amostras = 30
            
            while contador < meta_amostras:
                ret, frame = camera.read()
                if not ret:
                    break
                
                # Processar frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                # Desenhar informações
                cv2.putText(frame, f"Letra: {item}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Amostras: {contador}/{meta_amostras}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, "ESPACO: Capturar | ESC: Pular | Q: Sair", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Desenhar landmarks se detectados
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                cv2.imshow('TraduLibras - Coleta de Gestos', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Espaço - capturar
                    if results.multi_hand_landmarks:
                        landmarks = self.processar_landmarks(results.multi_hand_landmarks[0])
                        if len(landmarks) == 63:  # 21 pontos × 3 coordenadas
                            dados_coletados.append({
                                'label': item,
                                **{f'point_{i}': landmarks[i] for i in range(63)}
                            })
                            contador += 1
                            print(f"✅ Amostra {contador} capturada para {item}")
                        else:
                            print("⚠️ Landmarks inválidos, tente novamente")
                    else:
                        print("⚠️ Mão não detectada, tente novamente")
                
                elif key == 27:  # ESC - pular
                    print(f"⏭️ Pulando {item}")
                    break
                
                elif key == ord('q'):  # Q - sair
                    print("🚪 Saindo da coleta...")
                    camera.release()
                    cv2.destroyAllWindows()
                    return dados_coletados
        
        camera.release()
        cv2.destroyAllWindows()
        return dados_coletados
    
    def salvar_dados(self, novos_dados, nome_arquivo):
        """Salva dados coletados no arquivo CSV"""
        try:
            # Combinar dados existentes com novos
            todos_dados = self.dados_existentes + novos_dados
            
            # Criar DataFrame
            df = pd.DataFrame(todos_dados)
            
            # Salvar
            df.to_csv(nome_arquivo, index=False)
            
            print(f"✅ Dados salvos em {nome_arquivo}")
            print(f"📊 Total de amostras: {len(todos_dados)}")
            print(f"📈 Novas amostras: {len(novos_dados)}")
            
            # Mostrar distribuição
            print("\n📊 Distribuição por classe:")
            print(df['label'].value_counts().sort_index())
            
        except Exception as e:
            print(f"❌ Erro ao salvar dados: {e}")
    
    def treinar_modelo_expandido(self):
        """Treina modelo com vocabulário expandido"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            import pickle
            
            print("\n🧠 Treinando modelo expandido...")
            
            # Carregar dados
            df = pd.read_csv('gestos_libras.csv')
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
                'vocabulary_type': 'expanded'
            }
            
            with open('modelos/modelo_info_expandido.pkl', 'wb') as f:
                pickle.dump(model_info, f)
            
            print("✅ Modelo expandido salvo!")
            return True
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            return False
    
    def menu_principal(self):
        """Menu principal do expansor de vocabulário"""
        while True:
            print("\n" + "="*60)
            print("🎯 TraduLibras - Expansor de Vocabulário")
            print("="*60)
            print("1. 📝 Coletar letras faltantes (D-Z)")
            print("2. 🔢 Coletar números (0-9)")
            print("3. 📚 Coletar vocabulário completo (letras + números)")
            print("4. 🧠 Treinar modelo expandido")
            print("5. 📊 Ver estatísticas atuais")
            print("6. 🚪 Sair")
            print("="*60)
            
            opcao = input("Escolha uma opção (1-6): ").strip()
            
            if opcao == '1':
                dados = self.coletar_gestos(self.letras_para_adicionar, 'gestos_libras.csv')
                if dados:
                    self.salvar_dados(dados, 'gestos_libras.csv')
            
            elif opcao == '2':
                dados = self.coletar_gestos(self.numeros, 'gestos_libras.csv')
                if dados:
                    self.salvar_dados(dados, 'gestos_libras.csv')
            
            elif opcao == '3':
                vocabulario_completo = self.letras_para_adicionar + self.numeros
                dados = self.coletar_gestos(vocabulario_completo, 'gestos_libras.csv')
                if dados:
                    self.salvar_dados(dados, 'gestos_libras.csv')
            
            elif opcao == '4':
                self.treinar_modelo_expandido()
            
            elif opcao == '5':
                self.mostrar_estatisticas()
            
            elif opcao == '6':
                print("👋 Até logo!")
                break
            
            else:
                print("❌ Opção inválida!")

    def mostrar_estatisticas(self):
        """Mostra estatísticas dos dados atuais"""
        try:
            if os.path.exists('gestos_libras.csv'):
                df = pd.read_csv('gestos_libras.csv')
                print(f"\n📊 Estatísticas dos Dados:")
                print(f"📈 Total de amostras: {len(df)}")
                print(f"🏷️ Classes únicas: {len(df['label'].unique())}")
                print(f"📋 Classes: {sorted(df['label'].unique())}")
                print(f"\n📊 Distribuição:")
                print(df['label'].value_counts().sort_index())
            else:
                print("❌ Arquivo gestos_libras.csv não encontrado")
        except Exception as e:
            print(f"❌ Erro ao mostrar estatísticas: {e}")

if __name__ == "__main__":
    print("🚀 TraduLibras - Expansor de Vocabulário v1.0")
    print("📚 Este script permite expandir o vocabulário do reconhecimento")
    
    expansor = VocabularioExpansor()
    expansor.menu_principal()

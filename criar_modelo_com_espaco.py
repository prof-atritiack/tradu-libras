#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para criar modelo simples com gesto de ESPAÇO
Treina modelo com 21 letras + espaço = 22 classes
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Configurações do MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def process_landmarks(hand_landmarks):
    """Process hand landmarks and normalize relative to wrist (landmark 0) - Simple version"""
    if not hand_landmarks:
        return None

    # Ponto de referência (pulso)
    wrist = hand_landmarks.landmark[0]

    # Extrair apenas coordenadas x,y relativas ao pulso (42 features)
    features = []
    for landmark in hand_landmarks.landmarks:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y
        ])

    return features

def collect_gesture_data(gesture_name, samples_per_gesture=30):
    """Coleta dados de um gesto específico"""
    print(f"🎯 TREINAMENTO DO GESTO: {gesture_name}")
    print("=" * 50)
    print("📋 INSTRUÇÕES:")
    print(f"1. Faça o gesto de {gesture_name} em LIBRAS")
    print("2. Mantenha o gesto estável por alguns segundos")
    print("3. Pressione ESPAÇO para capturar")
    print("4. Pressione 'q' para sair")
    print("=" * 50)
    
    # Inicializar MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Inicializar câmera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Erro: Não foi possível abrir a câmera")
        return []
    
    print("✅ Câmera inicializada")
    
    # Lista para armazenar dados
    gesture_data = []
    sample_count = 0
    
    print(f"📊 Coletando {samples_per_gesture} amostras do gesto {gesture_name}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip horizontal para espelho
        frame = cv2.flip(frame, 1)
        
        # Converter BGR para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar com MediaPipe
        results = hands.process(rgb_frame)
        
        # Desenhar landmarks se detectados
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Informações na tela
        cv2.putText(frame, f"GESTO: {gesture_name} - Amostras: {sample_count}/{samples_per_gesture}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Pressione ESPACO para capturar", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Pressione 'q' para sair", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow(f'Treinamento - {gesture_name}', frame)
        
        # Capturar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Espaço para capturar
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                features = process_landmarks(landmarks)
                
                if features and len(features) == 42:
                    gesture_data.append(features)
                    sample_count += 1
                    print(f"✅ Amostra {sample_count} capturada!")
                    
                    if sample_count >= samples_per_gesture:
                        print(f"🎉 Coleta de {gesture_name} concluída!")
                        break
                else:
                    print("❌ Erro: Não foi possível processar landmarks")
            else:
                print("❌ Erro: Mão não detectada")
        
        elif key == ord('q'):
            break
    
    # Limpar recursos
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    return gesture_data

def create_model_with_space():
    """Cria modelo completo com gesto de espaço"""
    print("🚀 CRIAÇÃO DO MODELO COM ESPAÇO")
    print("=" * 50)
    
    # Lista de gestos para treinar
    gestures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'ESPACO']
    
    all_data = []
    all_labels = []
    
    for gesture in gestures:
        print(f"\n📝 Treinando gesto: {gesture}")
        gesture_data = collect_gesture_data(gesture, samples_per_gesture=30)
        
        if gesture_data:
            all_data.extend(gesture_data)
            all_labels.extend([gesture] * len(gesture_data))
            print(f"✅ {gesture}: {len(gesture_data)} amostras coletadas")
        else:
            print(f"❌ {gesture}: Nenhuma amostra coletada")
    
    if not all_data:
        print("❌ Nenhum dado coletado!")
        return
    
    print(f"\n📊 Total de amostras coletadas: {len(all_data)}")
    print(f"📊 Classes: {set(all_labels)}")
    
    # Converter para arrays numpy
    X = np.array(all_data)
    y = np.array(all_labels)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Dados de treino: {len(X_train)} amostras")
    print(f"📊 Dados de teste: {len(X_test)} amostras")
    
    # Treinar modelo
    print("\n🤖 Treinando modelo...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Avaliar modelo
    accuracy = model.score(X_test, y_test)
    print(f"🎯 Acurácia: {accuracy:.2%}")
    
    # Obter classes únicas
    classes = sorted(list(set(y)))
    print(f"📊 Classes finais: {classes}")
    
    # Salvar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_filename = f'modelos/modelo_com_espaco_{timestamp}.pkl'
    info_filename = f'modelos/modelo_info_com_espaco_{timestamp}.pkl'
    
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    model_info = {
        'classes': classes,
        'num_classes': len(classes),
        'features': 42,
        'accuracy': accuracy,
        'timestamp': timestamp,
        'description': 'Modelo simples com gesto de espaço'
    }
    
    with open(info_filename, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"\n✅ Modelo salvo: {model_filename}")
    print(f"✅ Info salva: {info_filename}")
    
    # Salvar dados
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(42)])
    df['label'] = y
    df.to_csv('gestos_com_espaco.csv', index=False)
    
    print(f"✅ Dados salvos: gestos_com_espaco.csv")
    
    return model_filename, info_filename

if __name__ == "__main__":
    model_file, info_file = create_model_with_space()
    
    if model_file and info_file:
        print("\n🎉 TREINAMENTO CONCLUÍDO!")
        print("=" * 50)
        print(f"📁 Modelo: {model_file}")
        print(f"📁 Info: {info_file}")
        print("🔄 Agora você pode atualizar o app.py para usar o novo modelo")
    else:
        print("\n❌ TREINAMENTO FALHOU!")

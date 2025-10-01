#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para treinar gesto de ESPAÇO em LIBRAS
Adiciona o gesto de espaço ao modelo existente
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import os

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
    for landmark in hand_landmarks.landmark:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y
        ])

    return features

def collect_space_gesture_data():
    """Coleta dados do gesto de espaço"""
    print("🎯 TREINAMENTO DO GESTO DE ESPAÇO")
    print("=" * 50)
    print("📋 INSTRUÇÕES:")
    print("1. Faça o gesto de ESPAÇO em LIBRAS")
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
    space_data = []
    sample_count = 0
    target_samples = 50  # 50 amostras do gesto de espaço
    
    print(f"📊 Coletando {target_samples} amostras do gesto de ESPAÇO...")
    
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
        cv2.putText(frame, f"GESTO DE ESPACO - Amostras: {sample_count}/{target_samples}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Pressione ESPACO para capturar", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Pressione 'q' para sair", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow('Treinamento - Gesto de ESPACO', frame)
        
        # Capturar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Espaço para capturar
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                features = process_landmarks(landmarks)
                
                if features and len(features) == 42:
                    space_data.append(features)
                    sample_count += 1
                    print(f"✅ Amostra {sample_count} capturada!")
                    
                    if sample_count >= target_samples:
                        print("🎉 Coleta de dados concluída!")
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
    
    return space_data

def update_model_with_space():
    """Atualiza o modelo existente com o gesto de espaço"""
    print("\n🔄 ATUALIZANDO MODELO COM GESTO DE ESPAÇO")
    print("=" * 50)
    
    # Carregar modelo existente
    try:
        with open('modelos/modelo_simples_20251001_112056.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('modelos/modelo_info_simples_20251001_112056.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        print("✅ Modelo existente carregado")
        print(f"📊 Classes atuais: {model_info['classes']}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return
    
    # Coletar dados do gesto de espaço
    space_data = collect_space_gesture_data()
    
    if not space_data:
        print("❌ Nenhum dado coletado")
        return
    
    print(f"📊 Coletadas {len(space_data)} amostras do gesto de ESPAÇO")
    
    # Preparar dados para treinamento
    X_space = np.array(space_data)
    y_space = ['ESPACO'] * len(space_data)
    
    # Carregar dados existentes
    try:
        df_existing = pd.read_csv('gestos_libras_expandido.csv')
        X_existing = df_existing.drop('label', axis=1).values
        y_existing = df_existing['label'].values
        
        print(f"📊 Dados existentes: {len(X_existing)} amostras")
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados existentes: {e}")
        return
    
    # Combinar dados
    X_combined = np.vstack([X_existing, X_space])
    y_combined = np.concatenate([y_existing, y_space])
    
    print(f"📊 Dados combinados: {len(X_combined)} amostras")
    
    # Treinar novo modelo
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    print("🤖 Treinando modelo atualizado...")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )
    
    # Treinar modelo
    model_updated = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model_updated.fit(X_train, y_train)
    
    # Avaliar modelo
    accuracy = model_updated.score(X_test, y_test)
    print(f"🎯 Acurácia do modelo atualizado: {accuracy:.2%}")
    
    # Obter classes únicas
    classes = sorted(list(set(y_combined)))
    print(f"📊 Classes finais: {classes}")
    
    # Salvar modelo atualizado
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_filename = f'modelos/modelo_com_espaco_{timestamp}.pkl'
    info_filename = f'modelos/modelo_info_com_espaco_{timestamp}.pkl'
    
    with open(model_filename, 'wb') as f:
        pickle.dump(model_updated, f)
    
    model_info_updated = {
        'classes': classes,
        'num_classes': len(classes),
        'features': 42,
        'accuracy': accuracy,
        'timestamp': timestamp,
        'description': 'Modelo simples com gesto de espaço'
    }
    
    with open(info_filename, 'wb') as f:
        pickle.dump(model_info_updated, f)
    
    print(f"✅ Modelo salvo: {model_filename}")
    print(f"✅ Info salva: {info_filename}")
    
    # Salvar dados atualizados
    df_updated = pd.DataFrame(X_combined, columns=[f'feature_{i}' for i in range(42)])
    df_updated['label'] = y_combined
    df_updated.to_csv('gestos_libras_com_espaco.csv', index=False)
    
    print(f"✅ Dados salvos: gestos_libras_com_espaco.csv")
    
    return model_filename, info_filename

if __name__ == "__main__":
    print("🚀 TREINAMENTO DO GESTO DE ESPAÇO")
    print("=" * 50)
    
    model_file, info_file = update_model_with_space()
    
    if model_file and info_file:
        print("\n🎉 TREINAMENTO CONCLUÍDO!")
        print("=" * 50)
        print(f"📁 Modelo: {model_file}")
        print(f"📁 Info: {info_file}")
        print("🔄 Agora você pode atualizar o app.py para usar o novo modelo")
    else:
        print("\n❌ TREINAMENTO FALHOU!")

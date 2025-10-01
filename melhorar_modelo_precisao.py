#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para melhorar precisão do modelo focando em letras similares
Treina modelo com mais dados para letras confusas (A/E, C/O)
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
from sklearn.metrics import classification_report, confusion_matrix

# Configurações do MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def process_landmarks(hand_landmarks):
    """Process hand landmarks and normalize relative to wrist (landmark 0) - Enhanced version"""
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
    # Distâncias entre pontos importantes
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
    
    return features  # Total: 42 + 5 + 4 = 51 features

def collect_gesture_data_enhanced(gesture_name, samples_per_gesture=50):
    """Coleta dados de um gesto específico com instruções detalhadas"""
    print(f"🎯 TREINAMENTO APRIMORADO: {gesture_name}")
    print("=" * 60)
    
    # Instruções específicas para letras problemáticas
    instructions = {
        'A': "GESTO A: Punho fechado com polegar estendido para cima (como 'ok')",
        'E': "GESTO E: Todos os dedos estendidos e juntos (como 'pare')",
        'C': "GESTO C: Mão em formato de C (dedos curvados como uma concha)",
        'O': "GESTO O: Mão em formato de O (dedos curvados formando círculo fechado)"
    }
    
    instruction = instructions.get(gesture_name, f"Faça o gesto de {gesture_name} em LIBRAS")
    
    print("📋 INSTRUÇÕES ESPECÍFICAS:")
    print(f"   {instruction}")
    print("📋 COMANDOS:")
    print("   • Pressione ESPAÇO para capturar")
    print("   • Pressione 'q' para sair")
    print("   • Pressione 'r' para reiniciar contagem")
    print("=" * 60)
    
    # Inicializar MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,  # Maior precisão
        min_tracking_confidence=0.7
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
        cv2.putText(frame, instruction[:50] + "...", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "ESPACO=capturar | Q=sair | R=reiniciar", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Mostrar frame
        cv2.imshow(f'Treinamento Aprimorado - {gesture_name}', frame)
        
        # Capturar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Espaço para capturar
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                features = process_landmarks(landmarks)
                
                if features and len(features) == 51:
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
        
        elif key == ord('r'):  # Reiniciar contagem
            sample_count = 0
            gesture_data = []
            print("🔄 Contagem reiniciada!")
        
        elif key == ord('q'):
            break
    
    # Limpar recursos
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    return gesture_data

def create_enhanced_model():
    """Cria modelo aprimorado com foco em letras similares"""
    print("🚀 CRIAÇÃO DO MODELO APRIMORADO")
    print("=" * 60)
    
    # Lista de gestos com foco nas letras problemáticas
    gestures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
    
    # Mais amostras para letras problemáticas
    samples_per_gesture = {
        'A': 80,  # Mais amostras para A
        'E': 80,  # Mais amostras para E
        'C': 80,  # Mais amostras para C
        'O': 80,  # Mais amostras para O
        'ESPACO': 50
    }
    
    all_data = []
    all_labels = []
    
    for gesture in gestures:
        samples = samples_per_gesture.get(gesture, 40)  # 40 amostras para outras letras
        print(f"\n📝 Treinando gesto: {gesture} ({samples} amostras)")
        gesture_data = collect_gesture_data_enhanced(gesture, samples)
        
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
    
    # Treinar modelo com parâmetros otimizados
    print("\n🤖 Treinando modelo aprimorado...")
    model = RandomForestClassifier(
        n_estimators=200,  # Mais árvores
        random_state=42,
        max_depth=15,      # Profundidade maior
        min_samples_split=3,  # Menos amostras para split
        min_samples_leaf=1,   # Menos amostras por folha
        max_features='sqrt',  # Otimização de features
        class_weight='balanced'  # Balanceamento de classes
    )
    
    model.fit(X_train, y_train)
    
    # Avaliar modelo
    accuracy = model.score(X_test, y_test)
    print(f"🎯 Acurácia geral: {accuracy:.2%}")
    
    # Relatório detalhado
    y_pred = model.predict(X_test)
    print("\n📊 RELATÓRIO DETALHADO:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusão para letras problemáticas
    print("\n🔍 MATRIZ DE CONFUSÃO (Letras Problemáticas):")
    problem_letters = ['A', 'E', 'C', 'O']
    problem_mask = np.isin(y_test, problem_letters)
    if np.any(problem_mask):
        cm = confusion_matrix(y_test[problem_mask], y_pred[problem_mask], labels=problem_letters)
        print("Confusão entre A/E/C/O:")
        for i, letter in enumerate(problem_letters):
            print(f"{letter}: {cm[i]}")
    
    # Obter classes únicas
    classes = sorted(list(set(y)))
    print(f"\n📊 Classes finais: {classes}")
    
    # Salvar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_filename = f'modelos/modelo_aprimorado_{timestamp}.pkl'
    info_filename = f'modelos/modelo_info_aprimorado_{timestamp}.pkl'
    
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    model_info = {
        'classes': classes,
        'num_classes': len(classes),
        'features': 51,  # 51 features aprimoradas
        'accuracy': accuracy,
        'timestamp': timestamp,
        'description': 'Modelo aprimorado com foco em letras similares',
        'enhanced_features': True
    }
    
    with open(info_filename, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"\n✅ Modelo salvo: {model_filename}")
    print(f"✅ Info salva: {info_filename}")
    
    # Salvar dados
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(51)])
    df['label'] = y
    df.to_csv('gestos_aprimorados.csv', index=False)
    
    print(f"✅ Dados salvos: gestos_aprimorados.csv")
    
    return model_filename, info_filename

if __name__ == "__main__":
    model_file, info_file = create_enhanced_model()
    
    if model_file and info_file:
        print("\n🎉 TREINAMENTO APRIMORADO CONCLUÍDO!")
        print("=" * 60)
        print(f"📁 Modelo: {model_file}")
        print(f"📁 Info: {info_file}")
        print("🔄 Agora você pode atualizar o app.py para usar o novo modelo")
        print("💡 O modelo tem 51 features e foco especial em letras similares")
    else:
        print("\n❌ TREINAMENTO FALHOU!")

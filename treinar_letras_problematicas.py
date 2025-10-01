#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para treinar especificamente as letras problemáticas C/D e A/E
Foco em melhorar a precisão dessas letras similares
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

def process_landmarks_enhanced(hand_landmarks):
    """Process hand landmarks with enhanced features for similar letters"""
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
    
    # Features extras específicas para letras similares
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Distâncias entre dedos e pulso (normalizadas)
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
    
    # Features específicas para diferenciação A/E e C/D
    # Para A/E: posição relativa do polegar
    thumb_relative_x = thumb_tip.x - wrist.x
    thumb_relative_y = thumb_tip.y - wrist.y
    features.extend([thumb_relative_x, thumb_relative_y])
    
    # Para C/D: curvatura dos dedos (distância entre pontas e bases)
    index_base = hand_landmarks.landmark[5]
    middle_base = hand_landmarks.landmark[9]
    ring_base = hand_landmarks.landmark[13]
    pinky_base = hand_landmarks.landmark[17]
    
    features.extend([
        abs(index_tip.x - index_base.x) + abs(index_tip.y - index_base.y),
        abs(middle_tip.x - middle_base.x) + abs(middle_tip.y - middle_base.y),
        abs(ring_tip.x - ring_base.x) + abs(ring_tip.y - ring_base.y),
        abs(pinky_tip.x - pinky_base.x) + abs(pinky_tip.y - pinky_base.y)
    ])
    
    # Ângulo do polegar (importante para A vs E)
    thumb_angle = np.arctan2(thumb_tip.y - wrist.y, thumb_tip.x - wrist.x)
    features.append(thumb_angle)
    
    # Spread dos dedos (importante para C vs D)
    finger_spread = abs(index_tip.x - pinky_tip.x) + abs(index_tip.y - pinky_tip.y)
    features.append(finger_spread)
    
    return features  # Total: 42 + 5 + 4 + 2 + 4 + 1 + 1 = 59 features

def collect_problematic_letters_data():
    """Coleta dados específicos para letras problemáticas"""
    print("🎯 TREINAMENTO ESPECÍFICO PARA LETRAS PROBLEMÁTICAS")
    print("=" * 60)
    
    # Instruções específicas e detalhadas
    instructions = {
        'A': {
            'gesture': "GESTO A: Punho fechado com polegar estendido para CIMA (como 'ok' ou 'thumbs up')",
            'tips': "• Polegar deve estar bem estendido para cima\n• Outros dedos fechados\n• Mão em posição vertical"
        },
        'E': {
            'gesture': "GESTO E: Todos os dedos estendidos e JUNTOS (como 'pare' ou 'stop')",
            'tips': "• Todos os dedos estendidos\n• Dedos bem juntos\n• Mão em posição vertical"
        },
        'C': {
            'gesture': "GESTO C: Mão em formato de C ABERTO (dedos curvados como uma concha)",
            'tips': "• Dedos curvados formando C\n• Espaço entre dedos\n• Formato de concha"
        },
        'D': {
            'gesture': "GESTO D: Indicador e médio estendidos, outros fechados (como 'peace')",
            'tips': "• Apenas indicador e médio estendidos\n• Polegar, anelar e mindinho fechados\n• Formato de 'V'"
        }
    }
    
    # Mais amostras para letras problemáticas
    samples_per_gesture = 100  # 100 amostras para cada letra problemática
    
    all_data = []
    all_labels = []
    
    for letter, info in instructions.items():
        print(f"\n📝 TREINANDO: {letter}")
        print("=" * 40)
        print(f"🎯 {info['gesture']}")
        print(f"💡 DICAS:\n{info['tips']}")
        print("=" * 40)
        
        gesture_data = collect_gesture_data_detailed(letter, samples_per_gesture, info)
        
        if gesture_data:
            all_data.extend(gesture_data)
            all_labels.extend([letter] * len(gesture_data))
            print(f"✅ {letter}: {len(gesture_data)} amostras coletadas")
        else:
            print(f"❌ {letter}: Nenhuma amostra coletada")
    
    return all_data, all_labels

def collect_gesture_data_detailed(gesture_name, samples_per_gesture, info):
    """Coleta dados com instruções detalhadas"""
    print(f"📊 Coletando {samples_per_gesture} amostras do gesto {gesture_name}...")
    
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
        cv2.putText(frame, info['gesture'][:60] + "...", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "ESPACO=capturar | Q=sair | R=reiniciar", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Mostrar frame
        cv2.imshow(f'Treinamento Detalhado - {gesture_name}', frame)
        
        # Capturar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Espaço para capturar
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                features = process_landmarks_enhanced(landmarks)
                
                if features and len(features) == 59:
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

def create_problematic_letters_model():
    """Cria modelo específico para letras problemáticas"""
    print("🚀 CRIAÇÃO DO MODELO PARA LETRAS PROBLEMÁTICAS")
    print("=" * 60)
    
    # Coletar dados
    all_data, all_labels = collect_problematic_letters_data()
    
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
    
    # Treinar modelo com parâmetros otimizados para letras similares
    print("\n🤖 Treinando modelo especializado...")
    model = RandomForestClassifier(
        n_estimators=300,  # Mais árvores para melhor precisão
        random_state=42,
        max_depth=20,     # Profundidade maior
        min_samples_split=2,  # Menos amostras para split
        min_samples_leaf=1,   # Menos amostras por folha
        max_features='sqrt',  # Otimização de features
        class_weight='balanced',  # Balanceamento de classes
        bootstrap=True,
        oob_score=True
    )
    
    model.fit(X_train, y_train)
    
    # Avaliar modelo
    accuracy = model.score(X_test, y_test)
    oob_score = model.oob_score_
    print(f"🎯 Acurácia geral: {accuracy:.2%}")
    print(f"🎯 OOB Score: {oob_score:.2%}")
    
    # Relatório detalhado
    y_pred = model.predict(X_test)
    print("\n📊 RELATÓRIO DETALHADO:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusão
    print("\n🔍 MATRIZ DE CONFUSÃO:")
    cm = confusion_matrix(y_test, y_pred, labels=['A', 'E', 'C', 'D'])
    print("Confusão entre A/E/C/D:")
    for i, letter in enumerate(['A', 'E', 'C', 'D']):
        print(f"{letter}: {cm[i]}")
    
    # Obter classes únicas
    classes = sorted(list(set(y)))
    print(f"\n📊 Classes finais: {classes}")
    
    # Salvar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_filename = f'modelos/modelo_problematicas_{timestamp}.pkl'
    info_filename = f'modelos/modelo_info_problematicas_{timestamp}.pkl'
    
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    model_info = {
        'classes': classes,
        'num_classes': len(classes),
        'features': 59,  # 59 features especializadas
        'accuracy': accuracy,
        'oob_score': oob_score,
        'timestamp': timestamp,
        'description': 'Modelo especializado para letras problemáticas (A/E, C/D)',
        'enhanced_features': True,
        'problematic_letters': True
    }
    
    with open(info_filename, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"\n✅ Modelo salvo: {model_filename}")
    print(f"✅ Info salva: {info_filename}")
    
    # Salvar dados
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(59)])
    df['label'] = y
    df.to_csv('gestos_problematicas.csv', index=False)
    
    print(f"✅ Dados salvos: gestos_problematicas.csv")
    
    return model_filename, info_filename

if __name__ == "__main__":
    model_file, info_file = create_problematic_letters_model()
    
    if model_file and info_file:
        print("\n🎉 TREINAMENTO ESPECIALIZADO CONCLUÍDO!")
        print("=" * 60)
        print(f"📁 Modelo: {model_file}")
        print(f"📁 Info: {info_file}")
        print("🔄 Agora você pode atualizar o app.py para usar o novo modelo")
        print("💡 O modelo tem 59 features especializadas para letras similares")
    else:
        print("\n❌ TREINAMENTO FALHOU!")

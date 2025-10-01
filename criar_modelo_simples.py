#!/usr/bin/env python3
"""
Script para criar um modelo simples e eficiente para detecção de gestos básicos
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from datetime import datetime

# Configurações do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def process_landmarks_simple(hand_landmarks):
    """Processa landmarks de forma mais simples - apenas coordenadas relativas"""
    if not hand_landmarks:
        return None
    
    # Ponto de referência (pulso)
    wrist = hand_landmarks.landmark[0]
    
    # Extrair apenas coordenadas x,y relativas ao pulso
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y
        ])
    
    return features

def collect_training_data():
    """Coleta dados de treinamento interativamente"""
    print("🎯 COLETA DE DADOS PARA MODELO SIMPLES")
    print("=" * 50)
    print("Instruções:")
    print("1. Posicione sua mão na frente da câmera")
    print("2. Faça o gesto da letra")
    print("3. Pressione a tecla correspondente")
    print("4. Pressione 'q' para sair")
    print("=" * 50)
    
    # Letras básicas para treinar
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    # Inicializar câmera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Erro: Não foi possível abrir a câmera")
        return None
    
    data = []
    labels = []
    
    print(f"\n📝 Letras disponíveis: {', '.join(letters)}")
    print("Pressione a tecla da letra para coletar dados...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip horizontalmente
        frame = cv2.flip(frame, 1)
        
        # Converter para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar com MediaPipe
        results = hands.process(rgb_frame)
        
        # Desenhar landmarks se detectados
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                
                # Processar landmarks
                features = process_landmarks_simple(hand_landmarks)
                if features:
                    # Mostrar na tela
                    cv2.putText(frame, "Mao detectada - Pressione tecla da letra", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar instruções
        cv2.putText(frame, "Pressione tecla da letra ou 'q' para sair", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Coleta de Dados - Modelo Simples', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key >= ord('a') and key <= ord('z'):
            letter = chr(key).upper()
            if letter in letters and results.multi_hand_landmarks:
                # Coletar dados
                features = process_landmarks_simple(results.multi_hand_landmarks[0])
                if features:
                    data.append(features)
                    labels.append(letter)
                    print(f"✅ Coletado: {letter} (Total: {len(data)} amostras)")
        
        elif key >= ord('A') and key <= ord('Z'):
            letter = chr(key)
            if letter in letters and results.multi_hand_landmarks:
                # Coletar dados
                features = process_landmarks_simple(results.multi_hand_landmarks[0])
                if features:
                    data.append(features)
                    labels.append(letter)
                    print(f"✅ Coletado: {letter} (Total: {len(data)} amostras)")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(data) == 0:
        print("❌ Nenhum dado coletado!")
        return None
    
    print(f"\n📊 Dados coletados: {len(data)} amostras")
    
    # Criar DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels
    
    return df

def train_simple_model(df):
    """Treina modelo simples"""
    print("\n🤖 TREINANDO MODELO SIMPLES...")
    print("=" * 50)
    
    # Separar features e labels
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    print(f"📊 Dados: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"📝 Classes: {sorted(set(y))}")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo simples
    model = RandomForestClassifier(
        n_estimators=50,  # Menos árvores para ser mais rápido
        max_depth=10,     # Limitar profundidade
        random_state=42,
        n_jobs=-1
    )
    
    print("🔄 Treinando...")
    model.fit(X_train, y_train)
    
    # Avaliar
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Acurácia: {accuracy:.2%}")
    print("\n📋 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Salvar modelo
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Criar diretório se não existir
    os.makedirs('modelos', exist_ok=True)
    
    # Salvar modelo
    model_path = f'modelos/modelo_simples_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Salvar informações do modelo
    model_info = {
        'timestamp': timestamp,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'classes': sorted(set(y)),
        'accuracy': accuracy,
        'model_type': 'RandomForestClassifier',
        'parameters': {
            'n_estimators': 50,
            'max_depth': 10
        }
    }
    
    info_path = f'modelos/modelo_info_simples_{timestamp}.pkl'
    with open(info_path, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"\n💾 Modelo salvo: {model_path}")
    print(f"💾 Info salva: {info_path}")
    
    return model, model_info, model_path, info_path

def main():
    """Função principal"""
    print("🚀 CRIADOR DE MODELO SIMPLES PARA TRADULIBRAS")
    print("=" * 60)
    
    # Coletar dados
    df = collect_training_data()
    if df is None:
        return
    
    # Treinar modelo
    model, model_info, model_path, info_path = train_simple_model(df)
    
    print("\n🎉 MODELO SIMPLES CRIADO COM SUCESSO!")
    print("=" * 50)
    print(f"📁 Modelo: {model_path}")
    print(f"📁 Info: {info_path}")
    print(f"🎯 Acurácia: {model_info['accuracy']:.2%}")
    print(f"📊 Classes: {len(model_info['classes'])}")
    print("=" * 50)
    
    # Atualizar app.py para usar o novo modelo
    print("\n🔄 Para usar o novo modelo, atualize o app.py:")
    print(f"   modelo_libras_simples_{model_info['timestamp']}.pkl")
    print(f"   modelo_info_simples_{model_info['timestamp']}.pkl")

if __name__ == "__main__":
    main()

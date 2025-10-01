#!/usr/bin/env python3
"""
Teste simples do coletor de dados
"""

import cv2
import mediapipe as mp
import numpy as np

def teste_camera():
    print("🎯 Testando Coletor de Dados LIBRAS")
    print("=" * 40)
    
    # Inicializar MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
    
    # Tentar inicializar câmera
    camera = None
    for camera_index in [0, 1, 2]:
        try:
            print(f"🔍 Tentando câmera {camera_index}...")
            camera = cv2.VideoCapture(camera_index)
            if camera.isOpened():
                ret, test_frame = camera.read()
                if ret and test_frame is not None:
                    print(f"✅ Câmera {camera_index} funcionando!")
                    break
                else:
                    camera.release()
                    camera = None
        except Exception as e:
            print(f"❌ Erro câmera {camera_index}: {e}")
            if camera:
                camera.release()
                camera = None
    
    if camera is None:
        print("❌ Nenhuma câmera disponível!")
        return
    
    print("\n📋 INSTRUÇÕES:")
    print("• Faça gestos com a mão")
    print("• Pressione ESPAÇO para coletar")
    print("• Pressione ESC para sair")
    print("• Pressione 'q' para sair")
    
    amostras_coletadas = 0
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            # Flip horizontal
            frame = cv2.flip(frame, 1)
            
            # Converter para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processar com MediaPipe
            results = hands.process(rgb_frame)
            
            # Desenhar landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Interface
            cv2.putText(frame, f"AMOSTRAS: {amostras_coletadas}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "ESPACO: Coletar | ESC/q: Sair", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Teste Coletor LIBRAS', frame)
            
            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Espaço
                if results.multi_hand_landmarks:
                    amostras_coletadas += 1
                    print(f"✅ Amostra {amostras_coletadas} coletada!")
                else:
                    print("❌ Mão não detectada")
            elif key == 27 or key == ord('q'):  # ESC ou q
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro: {e}")
    finally:
        # Limpar recursos
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Teste finalizado! {amostras_coletadas} amostras coletadas")

if __name__ == "__main__":
    teste_camera()


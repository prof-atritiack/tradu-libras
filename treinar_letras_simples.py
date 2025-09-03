import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Frases para treinar
FRASES = [
    "Oi Conselho Britanico",
    "TraduLibras"
]

# Extrair letras únicas das frases (convertendo para maiúsculas)
letras = set(''.join(FRASES).upper().replace(' ', ''))
print("\nLetras que serão treinadas:")
print(sorted(letras))
print(f"Total de letras únicas: {len(letras)}")

# Configurações
AMOSTRAS_POR_LETRA = 30  # Número de amostras para cada letra

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

def extrair_caracteristicas(landmarks):
    """Extrai as características dos pontos de referência da mão"""
    p0 = landmarks.landmark[0]  # Ponto de referência do pulso
    dados = []
    for lm in landmarks.landmark:
        dados.extend([
            lm.x - p0.x,
            lm.y - p0.y,
            lm.z - p0.z
        ])
    return dados

def coletar_dados():
    """Coleta dados para todas as letras necessárias"""
    dados_treinamento = []
    labels = []
    
    # Criar diretório para o modelo se não existir
    if not os.path.exists('modelos'):
        os.makedirs('modelos')
    
    camera = cv2.VideoCapture(0)
    
    for letra in sorted(letras):
        amostras_coletadas = 0
        print(f"\n=== Coletando dados para a letra '{letra}' ===")
        print(f"Objetivo: {AMOSTRAS_POR_LETRA} amostras")
        print("Pressione 'ESPAÇO' para capturar uma amostra")
        print("Pressione 'ESC' para pular esta letra")
        
        while amostras_coletadas < AMOSTRAS_POR_LETRA:
            success, frame = camera.read()
            if not success:
                continue
            
            # Espelha o frame para uma experiência mais natural
            frame = cv2.flip(frame, 1)
            
            # Processa o frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Desenha as informações na tela
            cv2.putText(frame, f"Letra: {letra}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Amostras: {amostras_coletadas}/{AMOSTRAS_POR_LETRA}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Se detectou mão, desenha os pontos
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cv2.imshow('Coleta de Dados', frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == 32 and results.multi_hand_landmarks:  # ESPAÇO
                # Extrai características e salva
                for hand_landmarks in results.multi_hand_landmarks:
                    caracteristicas = extrair_caracteristicas(hand_landmarks)
                    if len(caracteristicas) == 63:  # 21 pontos * 3 coordenadas
                        dados_treinamento.append(caracteristicas)
                        labels.append(letra)
                        amostras_coletadas += 1
                        print(f"Amostra {amostras_coletadas} coletada!")
            
            if amostras_coletadas >= AMOSTRAS_POR_LETRA:
                print(f"Coleta para letra {letra} finalizada!")
                break
    
    camera.release()
    cv2.destroyAllWindows()
    
    return np.array(dados_treinamento), np.array(labels)

def treinar_modelo(X, y):
    """Treina o modelo com os dados coletados"""
    print("\nTreinando o modelo...")
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Cria e treina o modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Avalia o modelo
    acuracia = modelo.score(X_test, y_test)
    print(f"\nAcurácia do modelo: {acuracia:.2%}")
    
    return modelo

def main():
    print("=== Treinamento de Reconhecimento de Letras ===")
    print("\nFrases de referência:")
    for frase in FRASES:
        print(f"- {frase}")
    
    input("\nPressione ENTER para começar a coleta de dados...")
    
    # Coleta os dados
    X, y = coletar_dados()
    
    if len(X) > 0:
        # Treina o modelo
        modelo = treinar_modelo(X, y)
        
        # Salva o modelo
        with open('modelos/modelo_libras.pkl', 'wb') as f:
            pickle.dump(modelo, f)
        print("\nModelo salvo com sucesso em 'modelos/modelo_libras.pkl'")
        
        # Salva as letras treinadas
        with open('modelos/letras_treinadas.txt', 'w') as f:
            f.write(','.join(sorted(letras)))
        print("Letras treinadas salvas em 'modelos/letras_treinadas.txt'")
    else:
        print("\nNenhum dado coletado. O modelo não foi treinado.")

if __name__ == "__main__":
    main() 
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
from gtts import gTTS
import os
import tempfile
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Load the trained model
with open('modelos/modelo_libras.pkl', 'rb') as f:
    model = pickle.load(f)

# Load model info
with open('modelos/modelo_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

# Global variables for text formation
current_letter = ""
formed_text = ""
corrected_text = ""
last_prediction_time = datetime.now()
prediction_cooldown = 1.0  # seconds

def process_landmarks(hand_landmarks):
    """Process hand landmarks and normalize relative to wrist (landmark 0)"""
    p0 = hand_landmarks.landmark[0]  # Wrist reference point
    points = []
    for landmark in hand_landmarks.landmark:
        points.extend([
            landmark.x - p0.x,
            landmark.y - p0.y,
            landmark.z - p0.z
        ])
    return points



def generate_frames():
    camera = cv2.VideoCapture(0)
    global current_letter, formed_text, corrected_text, last_prediction_time
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Process landmarks and make prediction
                points = process_landmarks(hand_landmarks)
                if len(points) == 63:  # Ensure we have the right number of features
                    prediction = model.predict([points])
                    
                    # Update current letter with cooldown
                    current_time = datetime.now()
                    if (current_time - last_prediction_time).total_seconds() >= prediction_cooldown:
                        current_letter = prediction[0]
                        formed_text += current_letter
                        corrected_text = formed_text  # Simple correction for now
                        last_prediction_time = current_time
                
                # Draw prediction on frame
                cv2.putText(frame, f"Letra: {current_letter}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    global formed_text, corrected_text
    return jsonify({
        'current_letter': current_letter,
        'formed_text': formed_text,
        'corrected_text': corrected_text
    })

@app.route('/clear_text')
def clear_text():
    global formed_text, corrected_text, current_letter
    formed_text = ""
    corrected_text = ""
    current_letter = ""
    return jsonify({
        'status': 'success',
        'message': 'Texto limpo com sucesso',
        'formed_text': formed_text,
        'corrected_text': corrected_text
    })



@app.route('/letra_atual')
def letra_atual():
    global current_letter
    return jsonify({'letra': current_letter})

@app.route('/falar_texto', methods=['POST'])
def falar_texto():
    from flask import request, send_file
    global corrected_text, formed_text
    
    data = request.get_json()
    texto = data.get('texto', '') if data else ''
    
    # Usar formed_text se nenhum texto for fornecido
    if not texto:
        texto = corrected_text if corrected_text else formed_text
    
    if not texto or texto.strip() == "":
        return jsonify({'error': 'Nenhum texto para falar'})
    
    try:
        # Usar gTTS para gerar o áudio
        tts = gTTS(text=texto, lang='pt-br', slow=False)
        
        # Criar arquivo temporário único
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        temp_file = os.path.join(temp_dir, f'speech_{timestamp}.mp3')
        
        # Salvar áudio
        tts.save(temp_file)
        
        # Retornar o arquivo de áudio como resposta
        return send_file(temp_file, mimetype='audio/mpeg', as_attachment=False)
            
    except Exception as e:
        return jsonify({'error': f'Erro na síntese de voz: {str(e)}'})

@app.route('/status')
def status():
    """Rota para verificar o status da aplicação"""
    try:
        # Verificar se o modelo está carregado
        model_loaded = model is not None
        model_classes = model_info.get('classes', []) if model_info else []
        
        # Verificar se a câmera está disponível
        camera_available = False
        try:
            cap = cv2.VideoCapture(0)
            camera_available = cap.isOpened()
            cap.release()
        except:
            pass
        
        return jsonify({
            'status': 'online',
            'model_loaded': model_loaded,
            'model_classes': model_classes,
            'camera_available': camera_available,
            'speech_available': True,
            'voice': 'gTTS (Google Text-to-Speech)',
            'version': '2.0.0',
            'features': [
                'Reconhecimento de gestos em tempo real',
                'Síntese de voz integrada ao navegador',
                'Interface web responsiva',
                'Sistema de cooldown para estabilização',
                'Formação automática de palavras'
            ]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

if __name__ == '__main__':
    print("🚀 Iniciando TraduLibras v2.0.0...")
    print("📱 Acesse: http://localhost:5000")
    print("🎤 Voz: gTTS (Google Text-to-Speech)")
    print("🤖 Modelo: Random Forest")
    print("📊 Classes:", model_info.get('classes', []) if model_info else [])
    app.run(debug=True, host='0.0.0.0', port=5000) 
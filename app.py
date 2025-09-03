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
prediction_cooldown = 2.5  # seconds - aumentado para dar mais tempo entre detecções
letter_detected = False  # Flag para indicar se uma letra foi detectada recentemente

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
                        # Só atualiza se a predição for válida e diferente da anterior
                        predicted_letter = prediction[0]
                        if predicted_letter and predicted_letter.strip() and predicted_letter != current_letter:
                            current_letter = predicted_letter
                            formed_text += current_letter
                            corrected_text = formed_text  # Simple correction for now
                            last_prediction_time = current_time
                            letter_detected = True  # Marca que uma letra foi detectada
                
                # Draw prediction on frame
                letra_display = current_letter if current_letter and current_letter.strip() else "-"
                cv2.putText(frame, f"Letra: {letra_display}", (10, 50),
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
    global formed_text, corrected_text, current_letter, letter_detected
    formed_text = ""
    corrected_text = ""
    current_letter = ""  # Reset para string vazia
    letter_detected = False  # Reset da flag de detecção
    return jsonify({
        'status': 'success',
        'message': 'Texto limpo com sucesso',
        'formed_text': formed_text,
        'corrected_text': corrected_text,
        'current_letter': current_letter
    })

@app.route('/reset_detection')
def reset_detection():
    global letter_detected
    letter_detected = False
    return jsonify({'status': 'success', 'detectada': letter_detected})



@app.route('/letra_atual')
def letra_atual():
    global current_letter, letter_detected
    # Retorna hífen se não há letra detectada ou se está vazio
    letra_para_retornar = current_letter if current_letter and current_letter.strip() else "-"
    return jsonify({
        'letra': letra_para_retornar,
        'detectada': letter_detected
    })

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
        
        # Obter informações de rede
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            local_ip = "127.0.0.1"
        
        return jsonify({
            'status': 'online',
            'model_loaded': model_loaded,
            'model_classes': model_classes,
            'camera_available': camera_available,
            'speech_available': True,
            'voice': 'gTTS (Google Text-to-Speech)',
            'version': '2.0.0',
            'network_info': {
                'local_ip': local_ip,
                'port': 5000,
                'access_urls': [
                    f"http://localhost:5000",
                    f"http://127.0.0.1:5000",
                    f"http://{local_ip}:5000"
                ]
            },
            'features': [
                'Reconhecimento de gestos em tempo real',
                'Síntese de voz integrada ao navegador',
                'Interface web responsiva',
                'Sistema de cooldown para estabilização',
                'Formação automática de palavras',
                'Acesso via rede local'
            ]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@app.route('/network-info')
def network_info():
    """Rota para obter informações de rede"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        return jsonify({
            'local_ip': local_ip,
            'port': 5000,
            'access_urls': {
                'local': f"http://localhost:5000",
                'local_alt': f"http://127.0.0.1:5000",
                'network': f"http://{local_ip}:5000"
            },
            'instructions': {
                'mobile': f"Para acessar do celular: http://{local_ip}:5000",
                'tablet': f"Para acessar do tablet: http://{local_ip}:5000",
                'other_pc': f"Para acessar de outro computador: http://{local_ip}:5000"
            },
            'requirements': [
                "Todos os dispositivos devem estar na mesma rede Wi-Fi",
                "Firewall deve permitir conexões na porta 5000",
                "Use o endereço IP mostrado acima"
            ]
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'fallback': 'http://localhost:5000'
        })

if __name__ == '__main__':
    import socket
    
    # Obter o IP local da máquina
    def get_local_ip():
        try:
            # Conecta a um endereço externo para descobrir o IP local
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    local_ip = get_local_ip()
    
    print("🚀 Iniciando TraduLibras v2.0.0...")
    print("=" * 50)
    print("📱 ACESSO LOCAL:")
    print(f"   http://localhost:5000")
    print(f"   http://127.0.0.1:5000")
    print("=" * 50)
    print("🌐 ACESSO NA REDE LOCAL:")
    print(f"   http://{local_ip}:5000")
    print("=" * 50)
    print("📱 Para acessar de outros dispositivos:")
    print("   1. Certifique-se que estão na mesma rede Wi-Fi")
    print("   2. Use o endereço acima no navegador")
    print("   3. Exemplo: http://192.168.1.100:5000")
    print("=" * 50)
    print("🎤 Voz: gTTS (Google Text-to-Speech)")
    print("🤖 Modelo: Random Forest")
    print("📊 Classes:", model_info.get('classes', []) if model_info else [])
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 
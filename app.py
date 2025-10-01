from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

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
import subprocess
import json
from auth import user_manager, User

app = Flask(__name__)
app.secret_key = 'tradulibras_secret_key_2024'  # Chave secreta para sessões

# Configurar Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Por favor, faça login para acessar esta página.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return user_manager.get_user(user_id)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Load the trained model (using enhanced model with all letters)
with open('modelos/modelo_aprimorado_20251001_115726.pkl', 'rb') as f:
    model = pickle.load(f)

# Load model info
with open('modelos/modelo_info_aprimorado_20251001_115726.pkl', 'rb') as f:
    model_info = pickle.load(f)

# Global variables for text formation
current_letter = ""
formed_text = ""
corrected_text = ""
last_prediction_time = datetime.now()
prediction_cooldown = 1.0  # seconds - tempo entre impressões de letras
letter_detected = False  # Flag para indicar se uma letra foi detectada recentemente

# Sistema de estabilização de gestos
gesture_stabilization_time = 1.5  # segundos para estabilizar gesto
gesture_validation_count = 3  # número de detecções consecutivas necessárias
last_hand_detection_time = None
gesture_predictions = []  # Lista para armazenar predições recentes
current_gesture_candidate = None
gesture_stable_start_time = None

# Sistema de detecção sequencial (sem sair da mão)
sequential_detection_enabled = True  # Permitir detecção sequencial
last_gesture_change_time = None
gesture_change_cooldown = 2.0  # segundos entre mudanças de gesto

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

def smart_postprocessing(predicted_letter):
    """Pós-processamento inteligente para reduzir erros"""
    global gesture_predictions, current_gesture_candidate, gesture_stable_start_time
    global last_gesture_change_time, current_letter
    
    current_time = datetime.now()
    
    # Verificar cooldown entre mudanças de gesto
    if last_gesture_change_time is not None:
        time_since_change = (current_time - last_gesture_change_time).total_seconds()
        if time_since_change < gesture_change_cooldown:
            return None
    
    # Adicionar predição à lista
    gesture_predictions.append({
        'letter': predicted_letter,
        'time': current_time
    })
    
    # Manter apenas as últimas 8 predições para análise mais robusta
    if len(gesture_predictions) > 8:
        gesture_predictions.pop(0)
    
    # Verificar se temos predições suficientes
    if len(gesture_predictions) < 5:
        return None
    
    # Análise inteligente das últimas predições
    recent_predictions = [p['letter'] for p in gesture_predictions[-5:]]
    
    # Contar frequência de cada letra
    from collections import Counter
    letter_counts = Counter(recent_predictions)
    most_common_letter = letter_counts.most_common(1)[0][0]
    most_common_count = letter_counts.most_common(1)[0][1]
    
    # Só aceitar se a letra mais comum aparecer pelo menos 3 vezes nas últimas 5 predições
    if most_common_count >= 3:
        # Verificar se é diferente da letra atual
        if most_common_letter != current_letter:
            # Validação adicional para letras problemáticas
            if validate_problematic_letters(most_common_letter, recent_predictions):
                # Verificar confiança da predição
                if enhance_prediction_confidence(most_common_letter, recent_predictions):
                    # Reset para próxima detecção
                    gesture_predictions.clear()
                    current_gesture_candidate = None
                    gesture_stable_start_time = None
                    last_gesture_change_time = current_time
                    return most_common_letter
    
    return None

def validate_problematic_letters(letter, recent_predictions):
    """Validação específica para letras problemáticas com análise de contexto"""
    # Para A/E: verificar se há confusão
    if letter in ['A', 'E']:
        # Se detectou A, verificar se não há muitos E nas predições
        if letter == 'A':
            e_count = recent_predictions.count('E')
            if e_count >= 2:  # Se há muitos E, pode ser erro
                return False
        # Se detectou E, verificar se não há muitos A nas predições
        elif letter == 'E':
            a_count = recent_predictions.count('A')
            if a_count >= 2:  # Se há muitos A, pode ser erro
                return False
    
    # Para C/D: verificar se há confusão
    if letter in ['C', 'D']:
        # Se detectou C, verificar se não há muitos D nas predições
        if letter == 'C':
            d_count = recent_predictions.count('D')
            if d_count >= 2:  # Se há muitos D, pode ser erro
                return False
        # Se detectou D, verificar se não há muitos C nas predições
        elif letter == 'D':
            c_count = recent_predictions.count('C')
            if c_count >= 2:  # Se há muitos C, pode ser erro
                return False
    
    # Para C/O: verificar se há confusão
    if letter in ['C', 'O']:
        # Se detectou C, verificar se não há muitos O nas predições
        if letter == 'C':
            o_count = recent_predictions.count('O')
            if o_count >= 2:  # Se há muitos O, pode ser erro
                return False
        # Se detectou O, verificar se não há muitos C nas predições
        elif letter == 'O':
            c_count = recent_predictions.count('C')
            if c_count >= 2:  # Se há muitos C, pode ser erro
                return False
    
    return True

def enhance_prediction_confidence(predicted_letter, recent_predictions):
    """Melhora a confiança da predição usando análise temporal"""
    # Contar ocorrências da letra nas últimas predições
    letter_count = recent_predictions.count(predicted_letter)
    total_predictions = len(recent_predictions)
    
    # Calcular confiança baseada na frequência
    confidence = letter_count / total_predictions
    
    # Só aceitar se confiança >= 60%
    return confidence >= 0.6



def generate_frames():
    # Tentar inicializar a câmera com diferentes índices
    camera = None
    camera_index = 0
    
    try:
        for camera_index in [0, 1, 2]:
            try:
                camera = cv2.VideoCapture(camera_index)
                if camera.isOpened():
                    # Testar se consegue ler um frame
                    ret, test_frame = camera.read()
                    if ret and test_frame is not None:
                        print(f"✅ Câmera inicializada com índice {camera_index}")
                        break
                    else:
                        camera.release()
                        camera = None
            except Exception as e:
                print(f"❌ Erro ao inicializar câmera {camera_index}: {e}")
                if camera:
                    camera.release()
                camera = None
        
        if camera is None:
            print("❌ Nenhuma câmera disponível!")
            # Retornar frame de erro
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "CAMERA NAO DISPONIVEL", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return
    except Exception as e:
        print(f"❌ Erro crítico na inicialização da câmera: {e}")
        return
    
    global current_letter, formed_text, corrected_text, last_prediction_time
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("❌ Erro ao ler frame da câmera")
                break
            
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                global last_hand_detection_time
                last_hand_detection_time = datetime.now()
                
                # Detecção sequencial - sempre processar quando mão detectada
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Process landmarks and make prediction
                    points = process_landmarks(hand_landmarks)
                    if points and len(points) == 51:  # Ensure we have the right number of features (51 for enhanced model)
                        try:
                            prediction = model.predict([points])
                            predicted_letter = prediction[0]
                            
                            # Usar pós-processamento inteligente
                            validated_letter = smart_postprocessing(predicted_letter)
                            
                            if validated_letter:
                                current_letter = validated_letter
                                
                                # Se for ESPAÇO, adicionar espaço ao texto
                                if validated_letter == 'ESPACO':
                                    formed_text += ' '
                                else:
                                    formed_text += validated_letter
                                
                                corrected_text = formed_text
                                letter_detected = True
                                print(f"✅ Letra validada detectada: {validated_letter}")
                            
                        except Exception as e:
                            print(f"❌ Erro na predição: {e}")
                            # Continue processing frames even if prediction fails
            
            # Convert frame to jpg
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    except Exception as e:
        print(f"❌ Erro no processamento de frames: {e}")
    finally:
        # Fechar câmera quando sair do loop
        if camera:
            camera.release()
            print("📷 Câmera liberada")

# Rotas de Autenticação
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = user_manager.authenticate(username, password)
        if user:
            login_user(user)
            flash(f'Bem-vindo, {user.username}!', 'success')
            
            # Redirecionar baseado no tipo de usuário
            if user.is_admin():
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('camera'))
        else:
            flash('Usuário ou senha incorretos!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logout realizado com sucesso!', 'info')
    return redirect(url_for('login'))

@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin():
        flash('Acesso negado! Apenas administradores podem acessar esta página.', 'error')
        return redirect(url_for('camera'))
    
    user_stats = user_manager.get_stats()
    return render_template('admin_dashboard.html', user_stats=user_stats)

# Rotas de gerenciamento de usuários (apenas admin)
@app.route('/admin/create-user', methods=['POST'])
@login_required
def create_user():
    if not current_user.is_admin():
        return jsonify({'success': False, 'error': 'Acesso negado'})
    
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')
    
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username e password são obrigatórios'})
    
    user = user_manager.create_user(username, password, role)
    if user:
        return jsonify({'success': True, 'message': 'Usuário criado com sucesso'})
    else:
        return jsonify({'success': False, 'error': 'Usuário já existe'})

@app.route('/admin/check-updates')
@login_required
def check_updates():
    if not current_user.is_admin():
        return jsonify({'error': 'Acesso negado'})
    
    try:
        # Verificar se há atualizações no GitHub
        result = subprocess.run(['git', 'fetch', 'origin'], capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'error': 'Erro ao verificar atualizações'})
        
        result = subprocess.run(['git', 'status', '-uno'], capture_output=True, text=True)
        updates_available = 'behind' in result.stdout
        
        return jsonify({
            'updates_available': updates_available,
            'message': 'Atualizações disponíveis' if updates_available else 'Sistema atualizado'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/admin/update-system', methods=['POST'])
@login_required
def update_system():
    if not current_user.is_admin():
        return jsonify({'success': False, 'error': 'Acesso negado'})
    
    try:
        # Fazer backup dos modelos
        import shutil
        from pathlib import Path
        
        backup_dir = Path('backup')
        backup_dir.mkdir(exist_ok=True)
        
        models_dir = Path('modelos')
        if models_dir.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for model_file in models_dir.glob('*.pkl'):
                backup_file = backup_dir / f'{model_file.stem}_backup_{timestamp}.pkl'
                shutil.copy2(model_file, backup_file)
        
        # Aplicar atualizações
        result = subprocess.run(['git', 'pull', 'origin', 'main'], capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({'success': False, 'error': 'Erro ao aplicar atualizações'})
        
        return jsonify({'success': True, 'message': 'Sistema atualizado com sucesso'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/create-backup', methods=['POST'])
@login_required
def create_backup():
    if not current_user.is_admin():
        return jsonify({'success': False, 'error': 'Acesso negado'})
    
    try:
        import shutil
        from pathlib import Path
        
        backup_dir = Path('backup')
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Backup dos modelos
        models_dir = Path('modelos')
        if models_dir.exists():
            for model_file in models_dir.glob('*.pkl'):
                backup_file = backup_dir / f'{model_file.stem}_backup_{timestamp}.pkl'
                shutil.copy2(model_file, backup_file)
        
        # Backup dos dados
        data_file = Path('gestos_libras.csv')
        if data_file.exists():
            backup_file = backup_dir / f'gestos_libras_backup_{timestamp}.csv'
            shutil.copy2(data_file, backup_file)
        
        # Backup dos usuários
        users_file = Path('users.json')
        if users_file.exists():
            backup_file = backup_dir / f'users_backup_{timestamp}.json'
            shutil.copy2(users_file, backup_file)
        
        return jsonify({'success': True, 'message': 'Backup criado com sucesso'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/clear-logs', methods=['POST'])
@login_required
def clear_logs():
    if not current_user.is_admin():
        return jsonify({'success': False, 'error': 'Acesso negado'})
    
    try:
        # Limpar logs (implementar conforme necessário)
        return jsonify({'success': True, 'message': 'Logs limpos com sucesso'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/logs')
@login_required
def view_logs():
    if not current_user.is_admin():
        flash('Acesso negado!', 'error')
        return redirect(url_for('camera'))
    
    # Implementar visualização de logs
    return render_template('logs.html')

@app.route('/admin/manage-users')
@login_required
def manage_users():
    if not current_user.is_admin():
        flash('Acesso negado!', 'error')
        return redirect(url_for('camera'))
    
    users = user_manager.list_users()
    return render_template('manage_users.html', users=users)

@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.is_admin():
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('camera'))
    return redirect(url_for('login'))

@app.route('/camera')
@login_required
def camera():
    return render_template('camera_tradulibras.html')

@app.route('/tutorial')
@login_required
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
    letra_para_retornar = "-"
    if current_letter and current_letter.strip() and current_letter.strip() != "":
        letra_para_retornar = current_letter.strip()
    
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
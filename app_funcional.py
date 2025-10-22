#!/usr/bin/env python3
"""
Versão FUNCIONAL e ESTÁVEL do TraduLibras
Sistema de reconhecimento INCLUSAO BC sem travamentos
"""

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, session, send_file
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
app.secret_key = 'tradulibras_secret_key_2024'

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

# Carregar modelo INCLUSAO BC
import glob
import os

# Caminho base dos modelos
pasta_modelos = 'modelos/'

# Busca todos os arquivos de modelo, scaler e info
modelos = sorted(glob.glob(os.path.join(pasta_modelos, 'modelo_libras_*.pkl')), key=os.path.getmtime)
scalers = sorted(glob.glob(os.path.join(pasta_modelos, 'scaler_libras_*.pkl')), key=os.path.getmtime)
infos = sorted(glob.glob(os.path.join(pasta_modelos, 'modelo_info_libras_*.pkl')), key=os.path.getmtime)

if modelos and scalers and infos:
    modelo_inclusao_bc = modelos[-1]
    scaler_inclusao_bc = scalers[-1]
    info_inclusao_bc = infos[-1]
else:
    raise FileNotFoundError("❌ Nenhum modelo LIBRAS encontrado na pasta 'modelos/'.")



if os.path.exists(modelo_inclusao_bc) and os.path.exists(scaler_inclusao_bc) and os.path.exists(info_inclusao_bc):
    with open(modelo_inclusao_bc, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_inclusao_bc, 'rb') as f:
        scaler = pickle.load(f)
    with open(info_inclusao_bc, 'rb') as f:
        model_info = pickle.load(f)
    print(f"📊 Classes: {model_info['classes']}")
else:
    model = None
    scaler = None
    model_info = {'classes': [], 'accuracy': 0}

# Variáveis globais
current_letter = ""
formed_text = ""
corrected_text = ""
last_prediction_time = datetime.now()
prediction_cooldown = 2.5
hand_detected_time = None
min_hand_time = 1.5
auto_speak_enabled = True  # 👈 NOVA VARIÁVEL

def process_landmarks(hand_landmarks):
    """Processar landmarks da mão"""
    if not hand_landmarks:
        return None
    
    # Ponto de referência (pulso)
    wrist = hand_landmarks.landmark[0]
    
    # Features básicas
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y
        ])
    
    # Features extras
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
    
    return features  # Total: 51 features

# ==================== DETECÇÃO AUTOMÁTICA DE WEBCAM USB ====================
def detectar_webcam_usb_automatico():
    """Detecta automaticamente a webcam USB (prioriza câmeras externas)""" 
    cameras_detectadas = []
    
    # Testa as primeiras 5 câmeras
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                success, frame = cap.read()
                if success:
                    # Obter informações da câmera
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    camera_info = {
                        'index': i,
                        'resolution': f"{width}x{height}",
                        'width': width,
                        'height': height,
                        'internal': i == 0,  # Considera câmera 0 como interna
                        'status': '✅ Disponível'
                    }
                    cameras_detectadas.append(camera_info)
                cap.release()
        except Exception as e:
            print(f"❌ Erro ao testar câmera {i}: {e}")
    
    # Priorizar câmeras USB (índice > 0)
    cameras_usb = [cam for cam in cameras_detectadas if not cam['internal']]
    cameras_internas = [cam for cam in cameras_detectadas if cam['internal']]
    
    if cameras_usb:
        # Usar a primeira webcam USB encontrada
        webcam_usb = cameras_usb[0]
        return webcam_usb['index']
    elif cameras_internas:
        # Fallback para câmera interna se não encontrar USB
        cam_interna = cameras_internas[0]
        return cam_interna['index']
    else:
        return 0  # Fallback para câmera 0

# Variável global para controlar a câmera selecionada - DETECTA AUTOMATICAMENTE
selected_camera_index = detectar_webcam_usb_automatico()

# ==================== FUNÇÃO DA CÂMERA ÚNICA E CORRIGIDA ====================
def generate_frames():
    """Gerar frames da câmera USB automaticamente detectada"""
    global current_letter, formed_text, corrected_text, last_prediction_time, hand_detected_time
    global selected_camera_index
    
    print(f"🎥 CONECTANDO AUTOMATICAMENTE NA WEBCAM {selected_camera_index}...")
    
    # Inicializar câmera selecionada automaticamente
    camera = cv2.VideoCapture(selected_camera_index)
    
    # Configurar câmera para melhor performance
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Verificar se a câmera abriu corretamente
    if not camera.isOpened():
        print(f"❌ ERRO: Não foi possível conectar na webcam {selected_camera_index}")
        print("💡 Tentando câmera padrão (0)...")
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("❌ Nenhuma câmera disponível")
            return
    
    # Obter informações finais da câmera
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ WEBCAM {selected_camera_index} CONECTADA COM SUCESSO!")
    print(f"   📐 Resolução: {actual_width}x{actual_height}")
    print(f"   🔄 Pronta para reconhecimento de LIBRAS")
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip horizontal
        frame = cv2.flip(frame, 1)
        
        # Converter para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar com MediaPipe
        results = hands.process(rgb_frame)
        
        # Variável para landmarks
        points = None
        current_time = datetime.now()
        
        # Interface visual LIMPA - apenas landmarks quando detecta mão
        if results.multi_hand_landmarks:
            # Se mão detectada pela primeira vez
            if hand_detected_time is None:
                hand_detected_time = current_time
                print("👋 Mão detectada! Aguardando estabilização...")
            
            # Desenhar landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Processar landmarks
                points = process_landmarks(hand_landmarks)
            
            # Mostrar status
            time_since_detection = (current_time - hand_detected_time).total_seconds()
            
            if time_since_detection < min_hand_time:
                # Aguardando estabilização - limpar letra atual
                current_letter = ""
            else:
                # Pronto para detectar
                time_since_last = (current_time - last_prediction_time).total_seconds()
                
                if time_since_last >= prediction_cooldown and points and len(points) == 51:
                    try:
                        if model and scaler:
                            # Normalizar features
                            points_normalized = scaler.transform([points])
                            prediction = model.predict(points_normalized)
                            predicted_letter = prediction[0]
                            
                            # Processar letra detectada
                            if predicted_letter == 'ESPACO':
                                current_letter = '[ESPAÇO]'
                                formed_text += ' '
                                corrected_text = formed_text
                            elif predicted_letter == '.':  # 👈 COMANDO PONTO CORRIGIDO
                                current_letter = '[PONTO]'
                                print("🎯 PONTO DETECTADO - Falando e limpando texto...")
                                
                                # Salvar o texto atual antes de limpar
                                texto_para_falar = formed_text.strip()
                                print(f"🎯 DEBUG: Texto para falar: '{texto_para_falar}'")
                                
                                # Limpar o texto primeiro
                                formed_text = ""
                                corrected_text = ""
                                print("🎯 DEBUG: Texto limpo")
                                
                                # Falar o texto salvo se não estiver vazio E se a fala automática estiver habilitada
                                if texto_para_falar and auto_speak_enabled:
                                    print("🎯 DEBUG: Iniciando thread de fala...")
                                    threading.Thread(target=falar_texto_automatico, args=(texto_para_falar,), daemon=True).start()
                                elif texto_para_falar and not auto_speak_enabled:
                                    print("🔇 Fala automática desativada - texto não falado")
                                else:
                                    print("🎯 DEBUG: Texto vazio - nada para falar")
                            else:
                                current_letter = predicted_letter
                                formed_text += predicted_letter
                                corrected_text = formed_text
                            
                            # Atualizar tempo da última predição
                            last_prediction_time = current_time
                            hand_detected_time = None
                            
                            print(f"✅ Letra detectada: {predicted_letter}")
                            
                    except Exception as e:
                        print(f"❌ Erro na predição: {e}")
        else:
            # Sem mão detectada - limpar letra atual
            hand_detected_time = None
            current_letter = ""
        
        # Interface COMPLETAMENTE LIMPA - sem textos na câmera
        
        # Converter frame para JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_jpeg = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')
    
    # Liberar câmera ao sair
    camera.release()
    print(f"🔴 Webcam {selected_camera_index} desconectada")

# ==================== FUNÇÃO DE FALA AUTOMÁTICA ====================
def falar_texto_automatico(texto_para_falar):
    """Fala o texto automaticamente usando PYGAME em segundo plano"""
    try:
        print(f"🎯 FALANDO COM PYGAME: '{texto_para_falar}'")
        
        if not texto_para_falar or not texto_para_falar.strip():
            print("⚠️  Texto vazio")
            return
            
        texto_limpo = texto_para_falar.strip()
        
        # Criar arquivo de áudio temporário
        tts = gTTS(text=texto_limpo, lang='pt-br')
        temp_file = os.path.join(tempfile.gettempdir(), f'pygame_fala_{int(time.time())}.mp3')
        tts.save(temp_file)
        
        print(f"📁 Áudio salvo: {temp_file}")
        
        # Reproduzir com pygame EM SEGUNDO PLANO
        try:
            import pygame
            
            # Inicializar pygame mixer (sem display)
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Carregar e reproduzir o áudio
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            print("🔊 Reproduzindo áudio em segundo plano...")
            
            # Esperar terminar de tocar (sem bloquear a thread principal)
            start_time = time.time()
            while pygame.mixer.music.get_busy():
                if time.time() - start_time > 30:  # Timeout de 30 segundos
                    print("⏰ Timeout - parando áudio")
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.1)
                
            print("✅ Áudio reproduzido com sucesso!")
            
            # Limpar recursos do pygame
            pygame.mixer.quit()
            
        except Exception as pygame_error:
            print(f"❌ Erro no pygame: {pygame_error}")
            # Fallback: método simples do sistema
            try:
                os.startfile(temp_file)
                print("🔄 Usando fallback do sistema")
            except Exception as fallback_error:
                print(f"❌ Fallback também falhou: {fallback_error}")
        
        # Limpar arquivo temporário
        threading.Thread(target=limpar_arquivo_temporario, args=(temp_file, 10)).start()
        
    except Exception as e:
        print(f"💥 ERRO na fala automática: {e}")
        import traceback
        traceback.print_exc()

def limpar_arquivo_temporario(arquivo, segundos):
    """Limpa arquivo temporário após um tempo"""
    time.sleep(segundos)
    try:
        if os.path.exists(arquivo):
            os.remove(arquivo)
            print(f"🧹 Arquivo temporário removido: {arquivo}")
    except Exception as e:
        print(f"❌ Erro ao remover arquivo temporário: {e}")

# ==================== ROTAS PARA TESTE DE FALA ====================
@app.route('/teste_fala/<texto>')
def teste_fala(texto):
    """Rota para testar a fala manualmente"""
    try:
        print(f"🎯 TESTE MANUAL: '{texto}'")
        threading.Thread(target=falar_texto_automatico, args=(texto,), daemon=True).start()
        return jsonify({"status": "teste_iniciado", "texto": texto})
    except Exception as e:
        return jsonify({"status": "erro", "error": str(e)})

@app.route('/teste_fala_direto/<texto>')
def teste_fala_direto(texto):
    """Teste direto sem thread"""
    try:
        print(f"🎯 TESTE DIRETO: '{texto}'")
        falar_texto_automatico(texto)
        return jsonify({"status": "sucesso", "mensagem": f"Texto processado: {texto}"})
    except Exception as e:
        return jsonify({"status": "erro", "mensagem": str(e)})

# ==================== ROTAS PARA CONTROLE DE FALA AUTOMÁTICA ====================
@app.route('/auto_speak/toggle', methods=['POST'])
@login_required
def toggle_auto_speak():
    """Ativa/desativa a fala automática"""
    global auto_speak_enabled
    
    data = request.get_json()
    if 'enabled' in data:
        auto_speak_enabled = data['enabled']
    
    return jsonify({
        'success': True,
        'auto_speak_enabled': auto_speak_enabled,
        'message': f'Fala automática {"ativada" if auto_speak_enabled else "desativada"}'
    })

@app.route('/auto_speak/status')
@login_required
def get_auto_speak_status():
    """Retorna status da fala automática"""
    return jsonify({
        'auto_speak_enabled': auto_speak_enabled
    })

# ==================== DIAGNÓSTICO SERIAL ====================
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
    print("✅ Biblioteca serial disponível")
except ImportError:
    SERIAL_AVAILABLE = False
    print("❌ Biblioteca serial não disponível")

def diagnosticar_portas_seriais():
    """Faz diagnóstico completo das portas seriais"""
    print("\n" + "="*60)
    print("🔍 DIAGNÓSTICO DE PORTAS SERIAIS")
    print("="*60)
    
    if not SERIAL_AVAILABLE:
        print("❌ Biblioteca pyserial não disponível")
        print("💡 Execute: pip install pyserial")
        return []
    
    try:
        ports = list(serial.tools.list_ports.comports())
        print(f"📋 Portas encontradas: {len(ports)}")
        
        if not ports:
            print("❌ Nenhuma porta serial detectada!")
            print("💡 Verifique se o Arduino está conectado via USB")
            return []
        
        portas_detalhadas = []
        
        for i, port in enumerate(ports, 1):
            print(f"\n{i}. {port.device}")
            print(f"   Descrição: {port.description}")
            print(f"   HWID: {port.hwid}")
            
            # Tentar detectar Arduino
            is_arduino = False
            arduino_types = []
            
            if 'arduino' in port.description.lower():
                is_arduino = True
                arduino_types.append("Descrição Arduino")
            if 'ch340' in port.description.lower():
                is_arduino = True
                arduino_types.append("CH340 (Clone Arduino)")
            if 'USB Serial' in port.description:
                is_arduino = True
                arduino_types.append("USB Serial")
            
            # Verificar se consegue abrir a porta
            try:
                teste = serial.Serial(port.device)
                teste.close()
                status = "✅ Disponível"
            except serial.SerialException as e:
                status = f"❌ Indisponível - {e}"
            
            if is_arduino:
                print(f"   🎯 ARDUINO DETECTADO: {', '.join(arduino_types)}")
            print(f"   📍 Status: {status}")
            
            port_info = {
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid,
                'is_arduino': is_arduino,
                'status': status,
                'arduino_types': arduino_types
            }
            portas_detalhadas.append(port_info)
        
        # Mostrar recomendações
        arduino_ports = [p for p in portas_detalhadas if p['is_arduino']]
        if arduino_ports:
            print(f"\n🎯 PORTAS ARDUINO RECOMENDADAS:")
            for port in arduino_ports:
                print(f"   → {port['device']} - {port['description']}")
        else:
            print("\n💡 DICAS:")
            print("   1. Conecte o Arduino via USB")
            print("   2. Instale drivers CH340 se necessário")
            print("   3. Reinicie o Arduino")
            print("   4. Feche a Arduino IDE")
        
        print("="*60)
        return portas_detalhadas
        
    except Exception as e:
        print(f"❌ Erro no diagnóstico: {e}")
        return []

# ==================== CONTROLE SERIAL SIMPLIFICADO ====================
class SerialController:
    def __init__(self):
        self.serial_connection = None
        self.port = None
        self.baudrate = 115200
        self.connected = False
        
    def list_ports(self):
        """Lista portas seriais de forma simples"""
        return diagnosticar_portas_seriais()
    
    def connect(self, port):
        """Conecta à porta serial"""
        if not SERIAL_AVAILABLE:
            return False, "Biblioteca serial não disponível"
        
        try:
            print(f"🔌 Tentando conectar em {port}...")
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=self.baudrate,
                timeout=1,
                write_timeout=1
            )
            time.sleep(2)  # Aguarda Arduino reiniciar
            self.port = port
            self.connected = True
            return True, f"Conectado à porta {port}"
        except serial.SerialException as e:
            error_msg = str(e)
            if "PermissionError" in error_msg or "acesso" in error_msg.lower():
                return False, f"Porta {port} em uso. Feche a Arduino IDE e outros programas."
            elif "FileNotFoundError" in error_msg:
                return False, f"Porta {port} não encontrada."
            else:
                return False, f"Erro: {error_msg}"
    
    def disconnect(self):
        """Desconecta da porta serial"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.connected = False
        self.port = None
        return True, "Desconectado"
    
    def send_letter(self, letter):
        """Envia uma letra para o Arduino"""
        if not self.connected or not self.serial_connection:
            return False, "Não conectado ao Arduino"
        
        try:
            letter = letter.lower().strip()
            if len(letter) == 1 and (letter.isalpha() or letter == '0'):
                self.serial_connection.write(letter.encode() + b'\n')
                self.serial_connection.flush()
                print(f"📤 Enviado: {letter.upper()}")
                return True, f"Letra '{letter.upper()}' enviada"
            else:
                return False, "Letra inválida"
        except Exception as e:
            return False, f"Erro ao enviar: {str(e)}"
    
    def get_status(self):
        """Retorna status da conexão serial"""
        return {
            'connected': self.connected,
            'port': self.port,
            'serial_available': SERIAL_AVAILABLE
        }

# Instância global do controlador serial
serial_controller = SerialController()

# ==================== ROTAS SERIAL ====================

@app.route('/serial/ports')
@login_required
def get_serial_ports():
    """Retorna lista de portas seriais"""
    ports = serial_controller.list_ports()
    return jsonify({'ports': ports})

@app.route('/serial/connect', methods=['POST'])
@login_required
def serial_connect():
    """Conecta à porta serial"""
    data = request.get_json()
    port = data.get('port')
    
    if not port:
        return jsonify({'success': False, 'message': 'Porta não especificada'})
    
    success, message = serial_controller.connect(port)
    return jsonify({'success': success, 'message': message})

@app.route('/serial/disconnect', methods=['POST'])
@login_required
def serial_disconnect():
    """Desconecta da porta serial"""
    success, message = serial_controller.disconnect()
    return jsonify({'success': success, 'message': message})

@app.route('/serial/status')
@login_required
def serial_status():
    """Retorna status da conexão serial"""
    status = serial_controller.get_status()
    return jsonify(status)

@app.route('/serial/send_letter', methods=['POST'])
@login_required
def send_serial_letter():
    """Envia uma letra para o Arduino"""
    data = request.get_json()
    letter = data.get('letter', '')
    
    if not letter:
        return jsonify({'success': False, 'message': 'Letra não especificada'})
    
    success, message = serial_controller.send_letter(letter)
    return jsonify({'success': success, 'message': message})

@app.route('/serial/send_word', methods=['POST'])
@login_required
def send_serial_word():
    """Envia uma palavra para o Arduino"""
    data = request.get_json()
    word = data.get('word', '')
    
    if not word:
        return jsonify({'success': False, 'message': 'Palavra não especificada'})
    
    if not serial_controller.connected:
        return jsonify({'success': False, 'message': 'Não conectado ao Arduino'})
    
    results = []
    for letter in word.lower():
        if letter.isalpha() or letter == ' ':
            if letter == ' ':
                time.sleep(1)
                results.append("Espaço - pausa")
            else:
                success, message = serial_controller.send_letter(letter)
                results.append(f"{letter.upper()}: {message}")
                time.sleep(0.8)
    
    return jsonify({'success': True, 'results': results})

# ==================== ROTA SIMPLES PARA STATUS DA CÂMERA ====================

@app.route('/camera/status')
@login_required
def get_camera_status():
    """Retorna status simples da câmera automática"""
    return jsonify({
        'camera_index': selected_camera_index,
        'status': 'conectada_automaticamente',
        'message': f'Webcam {selected_camera_index} conectada automaticamente'
    })

# ==================== ROTAS PRINCIPAIS (MANTIDAS) ====================

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('introducao'))  # ← Vai para introdução
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = user_manager.authenticate(username, password)
        
        if user:
            login_user(user)
            return redirect(url_for('introducao'))  # ← Vai para introdução
        else:
            flash('Usuário ou senha incorretos!')
    
    return render_template('login.html')

@app.route('/introducao')
@login_required
def introducao():
    """Tela de introdução usando template"""
    return render_template(
        'introducao.html',
        username=current_user.username,
        is_admin=current_user.is_admin()
    )

@app.route('/tutorial')
@login_required
def tutorial():
    """Página de tutorial do TRADULIBRAS"""
    return render_template('tutorial.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Você foi desconectado.')
    return redirect(url_for('login'))

@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin():
        flash('Acesso restrito a administradores!', 'error')
        return redirect(url_for('camera_tradulibras'))
    
    user_stats = user_manager.get_stats()
    return render_template('admin_dashboard.html', user_stats=user_stats)

@app.route('/camera')
@login_required
def camera_tradulibras():
    return render_template('camera_tradulibras.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==================== ROTAS PARA O BOTÃO APAGAR ====================

@app.route('/limpar_ultima_letra', methods=['POST'])
@login_required
def limpar_ultima_letra():
    global formed_text, current_letter
    
    print(f"📝 SOLICITAÇÃO: Apagar última letra")
    print(f"📝 Texto atual: '{formed_text}'")
    
    if formed_text:
        # Remove o último caractere
        formed_text = formed_text[:-1]
        current_letter = ""
        print(f"✅ NOVO TEXTO: '{formed_text}'")
        
        return jsonify({
            "status": "success", 
            "message": "Última letra removida", 
            "texto": formed_text
        })
    else:
        print("ℹ️ Nenhum texto para limpar")
        return jsonify({
            "status": "error", 
            "message": "Não há texto para limpar"
        })

@app.route('/letra_atual')
@login_required
def get_letra_atual():
    """Retorna a letra atual e texto formatado"""
    return jsonify({
        "letra": current_letter, 
        "texto": formed_text
    })

@app.route('/limpar_texto', methods=['POST'])
@login_required
def limpar_texto_completo():
    """Limpa todo o texto"""
    global formed_text, current_letter
    formed_text = ""
    current_letter = ""
    return jsonify({
        "status": "success", 
        "message": "Texto limpo completamente"
    })

@app.route('/falar_texto', methods=['GET', 'POST'])
@login_required
def falar_texto():
    """Fala o texto atual (manual)"""
    global formed_text
    
    if formed_text.strip():
        try:
            # Usar gTTS para gerar o áudio
            tts = gTTS(text=formed_text, lang='pt-br', slow=False)
            
            # Criar arquivo temporário único
            temp_dir = tempfile.gettempdir()
            timestamp = int(time.time())
            temp_file = os.path.join(temp_dir, f'manual_speech_{timestamp}.mp3')
            
            # Salvar áudio
            tts.save(temp_file)
            
            # Retornar o arquivo de áudio como resposta via send_file
            response = send_file(temp_file, mimetype='audio/mpeg', as_attachment=False)
            
            # Limpar arquivo após enviar (em thread separada)
            threading.Thread(target=limpar_arquivo_temporario, args=(temp_file, 30)).start()
            
            return response
            
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})
    
    return jsonify({"success": False, "error": "Nenhum texto para falar"})

@app.route('/status')
@login_required
def status():
    return jsonify({
        "modelo_carregado": model is not None,
        "classes": model_info.get('classes', []),
        "acuracia": model_info.get('accuracy', 0),
        "texto_atual": formed_text,
        "letra_atual": current_letter
    })

if __name__ == '__main__':
    print("🚀 TRADULIBRAS - WEBCAM USB AUTOMÁTICA")
    print("=" * 60)
    print(f"📊 Classes suportadas: {model_info.get('classes', [])}")
    print(f"📹 Webcam conectada: {selected_camera_index} (automático)")
    print(f"🎯 Acesso: http://localhost:5000")
    print("=" * 60)
    print("💡 CONECTANDO DIRETO NA WEBCAM USB...")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
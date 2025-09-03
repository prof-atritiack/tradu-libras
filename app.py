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
with open('modelo_libras.pkl', 'rb') as f:
    model = pickle.load(f)

# Global variables for text formation
current_letter = ""
formed_text = ""
corrected_text = ""
last_prediction_time = datetime.now()
prediction_cooldown = 1.0  # seconds

def process_landmarks(hand_landmarks):
    points = []
    for landmark in hand_landmarks.landmark:
        points.extend([landmark.x, landmark.y, landmark.z])
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
                prediction = model.predict([points])
                
                # Update current letter with cooldown
                current_time = datetime.now()
                if (current_time - last_prediction_time).total_seconds() >= prediction_cooldown:
                    current_letter = prediction[0]
                    formed_text += current_letter
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
    return jsonify({'status': 'success'})

@app.route('/text_to_speech')
def text_to_speech():
    global corrected_text
    if not corrected_text:
        return jsonify({'error': 'No text to speak'})
    
    try:
        tts = gTTS(text=corrected_text, lang='pt-br')
        
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, 'speech.mp3')
        
        # Save the audio file
        tts.save(temp_file)
        
        return jsonify({
            'status': 'success',
            'audio_path': temp_file
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 
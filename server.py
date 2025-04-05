from flask import Flask, jsonify, request
from flask_cors import CORS
import config
import base64
import json

app = Flask(__name__)
CORS(app)

@app.route('/analyze_images', methods=['POST'])
def analyze_images():
    data = request.get_json()
    
    # Store images in config
    if data.get('happy'):
        config.happy_face_image = data['happy']
    if data.get('serious'):
        config.serious_face_image = data['serious']
    if data.get('body'):
        config.body_image = data['body']
    config.gender = data.get('gender')

    # Here you would add your AI analysis logic
    # For now, setting dummy scores
    config.emotion_score = 85
    config.symmetry_score = 90
    config.body_score = 88
    
    return jsonify({
        'emotion_score': config.emotion_score,
        'symmetry_score': config.symmetry_score,
        'body_score': config.body_score,
        'error_msg': config.error_msg
    })

@app.route('/get_variables')
def get_variables():
    return jsonify({
        'emotion_score': config.emotion_score,
        'symmetry_score': config.symmetry_score,
        'body_score': config.body_score,
        'gender': config.gender,
        'error_msg': config.error_msg
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
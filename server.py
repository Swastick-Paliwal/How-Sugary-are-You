from flask import Flask, jsonify, request
from flask_cors import CORS
import config
import base64
import json
import emotion
import symmetry
import body
import cv2
from io import BytesIO
from PIL import Image
import numpy as np

def decode_base64_image(data_uri):
    header, encoded = data_uri.split(",", 1)
    binary_data = base64.b64decode(encoded)
    img = Image.open(BytesIO(binary_data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

app = Flask(__name__)
CORS(app)

@app.route('/analyze_images', methods=['POST'])
def analyze_images():
    data = request.get_json()
    
    # Store images in config
    if data.get('happy'):
        config.happy_face_image = decode_base64_image(data['happy'])
    if data.get('serious'):
        config.serious_face_image = decode_base64_image(data['serious'])
    if data.get('body'):
        config.body_image = decode_base64_image(data['body'])
    config.gender = data.get('gender')

    if config.happy_face_image is not None:
        emotion.main()  # Run emotion analysis
    if config.serious_face_image is not None:
        symmetry.main()  # Run symmetry analysis
    if config.body_image is not None:
        body.main() # Run body analysis

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
    app.run(debug=True, port=7000)
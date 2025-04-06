#  Sugahh Level for 418 Hackathon'25

Just as the thought, sight, or taste of sugar evokes a feeling of sweetness or happiness within us, a person who makes you feel similarly pleasant or joyful can metaphorically be referred to as your 'sugar.' In this context, our system evaluates and quantifies an individual's 'Sugahh Level' — a representation of their attractiveness or appeal to others. The ultimate objective of this project is to integrate this system into dating applications, enabling users to be matched with others who possess a comparable Sugahh Level, thereby promoting compatibility based on mutual attractiveness.


## Technical intro
A Python-based body & facial attractiveness analysis system that evaluates human attractiveness using real-time image analysis. The system combines body proportion analysis, facial emotion detection, and symmetry measurement to generate a final attractiveness score.

Built with:  
- Python  
- Flask  
- OpenCV  
- MediaPipe  
- DeepFace  
- face_recognition  

---

## Features

### 1. Body Analysis (`body.py`)
- Uses MediaPipe Pose for landmark detection.
- Calculates body proportions:
  - Shoulder-to-waist ratio
  - Waist-to-hip ratio
  - Leg-to-height ratio
  - Torso-to-leg ratio
- Checks symmetry between left and right body parts.
- Generates a body attractiveness score based on anthropometric ideals.

---

### 2. Facial Symmetry Analysis (`symmetry.py`)
- Analyzes a neutral or serious face image .
- Uses landmark-based symmetry measurement.
- `face_recognition` landmarks are taken and the flipped along the centre line of the face to determine symmetry

---

### 3. Facial Emotion Analysis (`emotion.py`)
- Detects emotions using DeepFace.
- Positive emotions (happy, neutral) increase score.
- Negative emotions (sad, angry, fear) reduce score.
- Final emotion score is a mix of emotion and symmetry of the smiling face
---

## Web Application

Frontend built using:  
- `index.htm` (HTML Structure)  
- `style.css` (Modern Styling with Blur Effects, Gradients, Google Fonts)  
- `script.js` (Handles Webcam Capture, Countdown Timer, AJAX Communication)

Backend powered by:  
- `server.py` (Flask Server)
  - Receives images and gender input.
  - Runs `body.py`, `emotion.py`, and `symmetry.py`.
  - Returns final scores.

---

## Installation & Setup  

### 1. Install Python dependencies:
```bash
pip install opencv-python mediapipe deepface face_recognition numpy flask
```

### 2. Run the Flask server:
```bash
python server.py
```

### 3. Open in Browser:
```
http://127.0.0.1:5000/
```

---

## Usage Workflow

1. Select Gender.
2. Capture 3 images:
   - Happy Face
   - Serious or Neutral Face
   - Full Body
3. Submit.
4. Get Results:
   - Body Score
   - Emotion Score
   - Symmetry Score
   - Final Attractiveness Score

---

## Customization  

| Parameter           | File        | Notes                         |
|--------------------|-------------|--------------------------------|
| Ideal Body Ratios  | `body.py`   | Modify constants for ratios.  |
| Emotion Weights    | `emotion.py`| Adjust weights in `main()`.   |
| Symmetry Threshold | `symmetry.py`| Change scoring logic.         |

---

## Error Handling
- All errors (e.g., face not detected) are stored in `config.error_msg`.

---

## Author  
Developed by [418 Pookies]  

---

## License  
This project is open-source under the MIT License.  

---
 Now then, gimme some sugahh honey ♥ so we can find the damn teapot.
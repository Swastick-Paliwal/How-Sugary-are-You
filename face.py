import cv2
from deepface import DeepFace
import face_recognition
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
image = frame if ret else None

# Analyze face
def get_face_score(image):
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        
        # Handle both list and dict formats
        if isinstance(result, list):
            emotions = result[0]['emotion']
        else:
            emotions = result['emotion']

        happy_score = emotions.get('happy', 0)
        neutral_score = emotions.get('neutral', 0)
        sad_score = emotions.get('sad', 0)

        # Simple proxy formula
        score = min((happy_score + 0.5 * neutral_score - sad_score), 100)
        print(f"Emotions: {emotions} => Score: {score}")
        
        return score
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 50  # fallback
def get_symmetry_score(image):
    try:
        face_landmarks_list = face_recognition.face_landmarks(image)
        if not face_landmarks_list:
            print("No face landmarks found.")
            return 50  # neutral fallback

        landmarks = face_landmarks_list[0]

        # Get key points around eyes, nose, and mouth
        keypoints = []
        for feature in ['left_eye', 'right_eye', 'nose_bridge', 'top_lip', 'bottom_lip']:
            keypoints.extend(landmarks.get(feature, []))

        keypoints = np.array(keypoints)
        x_coords = keypoints[:, 0]
        midline = np.mean([min(x_coords), max(x_coords)])

        # Flip x-coordinates over midline
        flipped_x = 2 * midline - keypoints[:, 0]
        flipped_points = np.column_stack((flipped_x, keypoints[:, 1]))

        # Measure distance between original and flipped
        symmetry_error = np.linalg.norm(keypoints - flipped_points, axis=1).mean()

        score = max(0, 100 - symmetry_error)  # Lower error means more symmetrical
        print(f"Symmetry error: {symmetry_error:.2f}, Score: {score:.2f}")
        return score
    except Exception as e:
        print(f"Error in symmetry score: {e}")
        return 50


emotion_score = get_face_score(image)
symmetry_score = get_symmetry_score(image)
face_score = 0 * emotion_score + 1 * symmetry_score


# Add score text and display
cv2.putText(frame, f"Attractiveness: {int(face_score)}%", (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
0
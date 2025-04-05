import cv2
import mediapipe as mp
import time
import numpy as np
import math
import config
import random
import traceback


# Set up MediaPipe Pose for body detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose=mp_pose.Pose(
    static_image_mode=True, 
    model_complexity=2,   # Highest accuracy model
    enable_segmentation=False, 
    min_detection_confidence=0.7  # You can even go higher like 0.8
)


# Define landmark indices for easier reference
NOSE = mp_pose.PoseLandmark.NOSE
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW
RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW
LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP
LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE
RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE
LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE
RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE

FULL_BODY_KEYPOINTS = [
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE
]

# Drawing parameters for beautified display
text_color = (240, 240, 240)
header_color = (50, 205, 50)
accent_color = (255, 215, 0)
error_color = (0, 0, 255)
box_color = (0, 0, 0)
box_alpha = 0.5

def calculate_distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks"""
    return math.sqrt(
        (landmark1.x - landmark2.x)**2 + 
        (landmark1.y - landmark2.y)**2
    )

def midpoint(landmark1, landmark2):
    """Calculate midpoint between two landmarks"""
    return (
        (landmark1.x + landmark2.x) / 2,
        (landmark1.y + landmark2.y) / 2
    )

def calculate_body_proportions(landmarks):
    """Calculate body proportions using end-to-end measurements"""
    
    # Shoulder width (end to end)
    shoulder_width = abs(landmarks[LEFT_SHOULDER].x - landmarks[RIGHT_SHOULDER].x)
    
    # Hip width (end to end)
    hip_width = abs(landmarks[LEFT_HIP].x - landmarks[RIGHT_HIP].x)
    
    # Waist position (end to end)
    waist_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2 - (
        ((landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2 - 
         (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2) * 0.4
    )
    left_waist_x = landmarks[LEFT_HIP].x
    right_waist_x = landmarks[RIGHT_HIP].x
    waist_width = abs(right_waist_x - left_waist_x)
    
    # Height measurements
    shoulder_top = min(landmarks[LEFT_SHOULDER].y, landmarks[RIGHT_SHOULDER].y)
    waist_bottom = waist_y
    ankle_bottom = max(landmarks[LEFT_ANKLE].y, landmarks[RIGHT_ANKLE].y)
    
    # Calculate vertical distances
    shoulder_to_waist = abs(waist_bottom - shoulder_top)
    waist_to_ankle = abs(ankle_bottom - waist_bottom)
    total_height = abs(ankle_bottom - shoulder_top)
    
    # Calculate proportions
    waist_to_hip_ratio = waist_width / hip_width if hip_width > 0 else 0
    shoulder_to_waist_ratio = shoulder_width / waist_width if waist_width > 0 else 0
    vertical_ratio = shoulder_to_waist / waist_to_ankle if waist_to_ankle > 0 else 0
    leg_to_height_ratio = waist_to_ankle / total_height if total_height > 0 else 0
    
    # Symmetry calculations (based on horizontal alignment)
    shoulder_symmetry = 1 - abs(landmarks[LEFT_SHOULDER].y - landmarks[RIGHT_SHOULDER].y) * 5
    hip_symmetry = 1 - abs(landmarks[LEFT_HIP].y - landmarks[RIGHT_HIP].y) * 5
    
    return {
        "waist_to_hip_ratio": waist_to_hip_ratio,
        "shoulder_to_waist_ratio": shoulder_to_waist_ratio,
        "leg_to_height_ratio": leg_to_height_ratio,
        "torso_to_leg_ratio": vertical_ratio,
        "shoulder_symmetry": shoulder_symmetry,
        "hip_symmetry": hip_symmetry
    }

def calculate_raw_score(proportions, ideal_props):
    """Calculate raw score based on deviations from ideal proportions"""
    total_deviation = sum(abs(proportions[k] - v) / v for k, v in ideal_props.items())
    raw_score = 100 * (1 - total_deviation)  # Convert to percentage
    return max(0, min(100, raw_score))  # Clamp between 0 and 100

def scale_score(raw_score):
    """Scale raw score to 50-100 range with nonlinear mapping"""
    # Apply sigmoid-like curve to spread scores
    scaled = 50 + (50 * math.tanh(raw_score / 50))
    return max(50, min(100, scaled))

def calculate_body_attractiveness(proportions):
    """Calculate body metrics with float precision"""
    if config.gender == "male":
        ideal_props = {
            "shoulder_to_waist_ratio": 1.8,  # Golden ratio
            "waist_to_hip_ratio": 0.95,
            "leg_to_height_ratio": 0.525,
            "torso_to_leg_ratio": 0.75
        }
    else:
        ideal_props = {
            "shoulder_to_waist_ratio": 1.4,
            "waist_to_hip_ratio": 0.75,
            "leg_to_height_ratio": 0.525,
            "torso_to_leg_ratio": 0.75
        }

    # Calculate scores directly from deviations
    scores = []
    for metric, ideal in ideal_props.items():
        actual = float(proportions[metric])
        deviation = abs(actual - ideal) / ideal
        
        # Convert deviation to score (100 = perfect match)
        if deviation <= 0.05:  # Within 5%
            score = 95.0 + (0.05 - deviation) * 100
        elif deviation <= 0.15:  # Within 15%
            score = 85.0 + (0.15 - deviation) * 100
        elif deviation <= 0.25:  # Within 25%
            score = 75.0 + (0.25 - deviation) * 50
        else:
            score = max(50.0, 75.0 - (deviation - 0.25) * 100)
        
        scores.append(score)
    
    # Add symmetry scores
    shoulder_sym = float(proportions["shoulder_symmetry"]) * 100
    hip_sym = float(proportions["hip_symmetry"]) * 100
    scores.extend([shoulder_sym, hip_sym])
    
    # Calculate weighted average with float precision
    weights = [0.3, 0.3, 0.2, 0.1, 0.05, 0.05]
    final_score = sum(float(s) * w for s, w in zip(scores, weights))
    
    # Add small variation while maintaining float precision
    variation = random.uniform(-1.0, 1.0)
    final_score = max(50.0, min(100.0, final_score + variation))
    
    return final_score

def analyze_image(frame):
    """Analyze a single image and return body metrics"""
    
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )
        
        landmarks = results.pose_landmarks.landmark
        full_body_visible = all(landmarks[kp].visibility > 0.5 for kp in FULL_BODY_KEYPOINTS)
        
        if full_body_visible:
            proportions = calculate_body_proportions(landmarks)
            body_attractiveness = calculate_body_attractiveness(proportions)
            
            # Draw results with taller box for 4 ratios
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (300, 210), box_color, -1)  # Increased height
            cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)
            
            # Header
            cv2.putText(frame, "BODY ANALYSIS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, header_color, 2)
            
            # Score
            score_color = (0, min(255, body_attractiveness * 2.55), 
                         min(255, (100 - body_attractiveness) * 2.55))
            cv2.putText(frame, f"Score: {body_attractiveness:.4f}%", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)
            
            # All four ratios
            cv2.putText(frame, f"Shoulder/Waist: {proportions['shoulder_to_waist_ratio']:.2f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            cv2.putText(frame, f"Waist/Hip: {proportions['waist_to_hip_ratio']:.2f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            cv2.putText(frame, f"Leg/Height: {proportions['leg_to_height_ratio']:.2f}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            cv2.putText(frame, f"Torso/Leg: {proportions['torso_to_leg_ratio']:.2f}", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            
            cv2.imshow('Body Analysis', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return body_attractiveness
        else:
            print("Full body not visible in image")
            return None
    else:
        print("No person detected in image")
        return None

def main():
    """Main function to run the body analysis program"""
    try:
        # from path
        # image_path = r"C:\Users\parvd\OneDrive\Desktop\test4.jpg"
        # frame = cv2.imread(image_path)

        # from webcam
        cap = cv2.VideoCapture(1)
        ret, frame = cap.read()
        cap.release()

        if not hasattr(config, 'gender') or config.gender not in ['male', 'female']:
            config.gender = 'female'
        if frame is None:
            raise Exception("Could not read the image")
        
        score = analyze_image(frame)
        if score:
            print(f"Body Score: {score}%")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
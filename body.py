import cv2
import mediapipe as mp
import time
import numpy as np
import math
import config
import random  # Import random for score variation
import traceback  # Import traceback for error handling

# Global variable for body metrics
config.body_attractiveness = 0
config.error_msg
config.gender


# Set up MediaPipe Pose for body detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

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

# List of key points that should be visible for a full body
FULL_BODY_KEYPOINTS = [
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE
]

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
    """Calculate various body proportion metrics"""
    
    # Get shoulder width
    shoulder_width = calculate_distance(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
    
    # Get hip width
    hip_width = calculate_distance(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
    
    # Calculate waist position (more accurate estimation)
    left_waist_y = landmarks[LEFT_HIP].y - (landmarks[LEFT_HIP].y - landmarks[LEFT_SHOULDER].y) * 0.6
    right_waist_y = landmarks[RIGHT_HIP].y - (landmarks[RIGHT_HIP].y - landmarks[RIGHT_SHOULDER].y) * 0.6
    left_waist_x = landmarks[LEFT_HIP].x - (landmarks[LEFT_HIP].x - landmarks[LEFT_SHOULDER].x) * 0.6
    right_waist_x = landmarks[RIGHT_HIP].x - (landmarks[RIGHT_HIP].x - landmarks[RIGHT_SHOULDER].x) * 0.6
    
    waist_width = math.sqrt((right_waist_x - left_waist_x)**2 + (right_waist_y - left_waist_y)**2)
    
    # Calculate torso height
    upper_torso = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2  # Average shoulder height
    lower_torso = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2  # Average hip height
    torso_height = lower_torso - upper_torso
    
    # Calculate leg length (average of both legs)
    left_leg_length = (calculate_distance(landmarks[LEFT_HIP], landmarks[LEFT_KNEE]) + 
                       calculate_distance(landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE]))
    right_leg_length = (calculate_distance(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE]) + 
                        calculate_distance(landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE]))
    leg_length = (left_leg_length + right_leg_length) / 2
    
    # Calculate full body height
    head_top = landmarks[NOSE].y - 0.15  # Estimate top of head
    ankle_bottom = (landmarks[LEFT_ANKLE].y + landmarks[RIGHT_ANKLE].y) / 2
    body_height = abs(ankle_bottom - head_top)
    
    # Calculate proportions
    waist_to_hip_ratio = waist_width / hip_width if hip_width > 0 else 0
    shoulder_to_waist_ratio = shoulder_width / waist_width if waist_width > 0 else 0
    shoulder_to_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 0
    leg_to_height_ratio = leg_length / body_height if body_height > 0 else 0
    torso_to_leg_ratio = torso_height / leg_length if leg_length > 0 else 0
    
    # Symmetry calculations
    shoulder_symmetry = 1 - abs(landmarks[LEFT_SHOULDER].y - landmarks[RIGHT_SHOULDER].y) * 5
    hip_symmetry = 1 - abs(landmarks[LEFT_HIP].y - landmarks[RIGHT_HIP].y) * 5
    
    return {
        "waist_to_hip_ratio": waist_to_hip_ratio,
        "shoulder_to_waist_ratio": shoulder_to_waist_ratio,
        "shoulder_to_hip_ratio": shoulder_to_hip_ratio,
        "leg_to_height_ratio": leg_to_height_ratio,
        "torso_to_leg_ratio": torso_to_leg_ratio,
        "shoulder_symmetry": shoulder_symmetry,
        "hip_symmetry": hip_symmetry
    }

def calculate_body_attractiveness(proportions):
    """Calculate enhanced body metrics using ratios based on anthropometric studies
    with optimistic scoring and gender-based logic."""
    
    scores = []
    
    # Shoulder to waist ratio (ideal ~1.618 for males, ~1.4 for females)
    if config.gender == "male":
        s2w_score = 100 - min(70, abs(proportions["shoulder_to_waist_ratio"] - 1.618) * 60)
    elif config.gender == "female":
        s2w_score = 100 - min(70, abs(proportions["shoulder_to_waist_ratio"] - 1.4) * 60)
    else:
        s2w_score = 0  # Default to 0 if gender is not set
    scores.append(s2w_score)
    
    # Waist to hip ratio (ideal ~0.7-0.8 for females, ~0.9-1.0 for males)
    if config.gender == "male":
        whr_score = 100 - min(70, abs(proportions["waist_to_hip_ratio"] - 0.95) * 120)
    elif config.gender == "female":
        whr_score = 100 - min(70, abs(proportions["waist_to_hip_ratio"] - 0.75) * 120)
    else:
        whr_score = 0
    scores.append(whr_score)
    
    # Leg to height ratio (ideal ~0.5-0.55)
    leg_score = 100 - min(70, abs(proportions["leg_to_height_ratio"] - 0.525) * 300)
    scores.append(leg_score)
    
    # Torso to leg ratio (ideal ~0.75 in many studies)
    torso_score = 100 - min(70, abs(proportions["torso_to_leg_ratio"] - 0.75) * 80)
    scores.append(torso_score)
    
    # Add symmetry scores
    shoulder_sym_score = min(100, proportions["shoulder_symmetry"] * 100)
    hip_sym_score = min(100, proportions["hip_symmetry"] * 100)
    scores.append(shoulder_sym_score)
    scores.append(hip_sym_score)
    
    # Calculate weighted average with bias toward higher scores
    raw_score = sum(scores) / len(scores)
    
    # Apply optimistic curve to increase scores
    optimistic_score = 60 + 40 * math.sqrt(raw_score / 100)
    
    # Introduce random variation in the score
    variation = random.uniform(-5, 5)  # Random variation between -5 and 5
    final_score = optimistic_score + variation
    
    return min(100, max(60, final_score))  # Ensure score is between 60 and 100

# Drawing parameters for beautified display
text_color = (240, 240, 240)
header_color = (50, 205, 50)
accent_color = (255, 215, 0)
error_color = (0, 0, 255)
box_color = (0, 0, 0)
box_alpha = 0.5

def analyze_image(image_path):
    """Analyze a single image and return body metrics"""
    frame = cv2.imread(image_path)
    if frame is None:
        raise Exception("Could not read the image")
    
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )
        
        landmarks = results.pose_landmarks.landmark
        full_body_visible = True
        
        for keypoint in FULL_BODY_KEYPOINTS:
            if landmarks[keypoint].visibility < 0.5:
                full_body_visible = False
                break
        
        if full_body_visible:
            proportions = calculate_body_proportions(landmarks)
            body_attractiveness = calculate_body_attractiveness(proportions)
            body_attractiveness = max(60, min(100, int(body_attractiveness)))
            
            # Draw results
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (300, 150), box_color, -1)
            cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)
            
            cv2.putText(frame, "BODY ANALYSIS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, header_color, 2)
            
            score_color = (0, min(255, body_attractiveness * 2.55), 
                         min(255, (100 - body_attractiveness) * 2.55))
            cv2.putText(frame, f"Score: {body_attractiveness}%", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)
            
            cv2.putText(frame, f"S/W Ratio: {proportions['shoulder_to_waist_ratio']:.2f}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            cv2.putText(frame, f"W/H Ratio: {proportions['waist_to_hip_ratio']:.2f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            
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

if __name__ == "__main__":
    try:
        image_path = r"C:\\Users\\parvd\\OneDrive\\Desktop\\test2.jpg"  # Replace with your image path
        score = analyze_image(image_path)
        if score:
            print(f"Body Score: {score}")
        else:
            print("Could not calculate body score")
    except Exception as e:
        print(f"Error analyzing image: {e}")
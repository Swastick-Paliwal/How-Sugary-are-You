import cv2
import mediapipe as mp
import time
import numpy as np
import math
import config
import random  # Import random for score variation

# Global variable for body metrics
config.body_attractiveness = 0
config.error_msg
config.gender

def try_camera_index(index):
    """Try to open camera at specified index"""
    print(f"Trying to open camera at index {index}...")
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Successfully opened camera at index {index}")
        return cap
    else:
        print(f"Failed to open camera at index {index}")
        return None

# Try to find an available camera
camera_found = False
cap = None

# Try different camera indices
for i in range(3):  # Try indices 0, 1, and 2
    cap = try_camera_index(i)
    if cap is not None:
        camera_found = True
        break

if not camera_found:
    print("Error: Could not open any camera. Please check your camera connection or permissions.")
    exit()

# Set up MediaPipe Pose for body detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

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

try:
    print("Camera opened successfully. Press 'q' to quit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
            
        # Process with MediaPipe to detect body
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Create overlay for text background
        overlay = frame.copy()
        
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )
            
            # Check if full body is visible
            landmarks = results.pose_landmarks.landmark
            full_body_visible = True
            missing_parts = []
            
            for keypoint in FULL_BODY_KEYPOINTS:
                if landmarks[keypoint].visibility < 0.5:  # Slightly more forgiving threshold
                    full_body_visible = False
                    missing_parts.append(keypoint.name)
            
            if full_body_visible:
                # Calculate body proportions
                proportions = calculate_body_proportions(landmarks)
                
                # Calculate body metrics based on proportions
                body_attractiveness = calculate_body_attractiveness(proportions)
                
                # Round the score to an integer
                body_attractiveness = max(60, min(100, int(body_attractiveness)))
                
                # Draw semi-transparent box for text background
                cv2.rectangle(overlay, (5, 5), (300, 150), box_color, -1)
                cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)
                
                # Display attractiveness score with enhanced visual style
                cv2.putText(frame, "BODY ANALYSIS", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, header_color, 2)
                
                # Score display with color based on score
                score_color = (0, min(255, body_attractiveness * 2.55), min(255, (100 - body_attractiveness) * 2.55))
                cv2.putText(frame, f"Score: {body_attractiveness}%", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)
                
                # Display key ratios
                cv2.putText(frame, f"Shoulder/Waist: {proportions['shoulder_to_waist_ratio']:.2f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                cv2.putText(frame, f"Waist/Hip: {proportions['waist_to_hip_ratio']:.2f}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                
            else:
                # Draw semi-transparent box for error text background
                cv2.rectangle(overlay, (5, 5), (300, 90), box_color, -1)
                cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)
                
                # Display error about body not fully visible
                cv2.putText(frame, "Body not fully visible", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, error_color, 2)
                # List missing parts (limited to avoid cluttering the screen)
                if missing_parts:
                    missing_text = f"Missing: {', '.join([p.split('.')[-1] for p in missing_parts[:3]])}"
                    if len(missing_parts) > 3:
                        missing_text += "..."
                    cv2.putText(frame, missing_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, error_color, 1)
        else:
            # Draw semi-transparent box for error text background
            cv2.rectangle(overlay, (5, 5), (300, 40), box_color, -1)
            cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)
            
            # No pose detected
            cv2.putText(frame, "No person detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, error_color, 2)
        
        # Display the processed image
        cv2.imshow('Body Analysis', frame)
        
        # Check for quit command (q key)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program...")
            break
            
        # Small delay to reduce CPU usage
        time.sleep(0.01)
            
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    # Release resources
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(f"Final Body Metrics: {body_attractiveness}")
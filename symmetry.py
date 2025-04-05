import cv2
from deepface import DeepFace
import face_recognition  
import numpy as np
import config
   
def linear_score(normalized_error, min_error=0.26, max_error=0.36):
    """
    Linearly maps normalized_error to a score between 50 and 100%.
    0.26 → 100%, 0.36 → 50%
    """
    normalized_error = float(normalized_error)
    slope = -50 / (max_error - min_error)
    score = slope * (normalized_error - min_error) + 100
    return max(50, min(score, 100))


def get_symmetry_score(image, frame=None, draw=True, curve_sharpness=8, curve_offset=0.02):
    try:
        face_landmarks_list = face_recognition.face_landmarks(image)
        if not face_landmarks_list:
            print("No face landmarks found.")
            config.error_msg = "No face landmarks found."
            return -1


        landmarks = face_landmarks_list[0]

        features = ['left_eye', 'right_eye', 'nose_bridge', 'top_lip', 'bottom_lip']
        keypoints = []

        for feature in features:
            points = landmarks.get(feature, [])
            keypoints.extend(points)

        keypoints = np.array(keypoints)
        x_coords = keypoints[:, 0]
        y_coords = keypoints[:, 1]

        midline = np.mean([min(x_coords), max(x_coords)])
        flipped_x = 2 * midline - x_coords
        flipped_points = np.column_stack((flipped_x, y_coords))

        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        normalization_factor = np.hypot(width, height)

        if normalization_factor == 0:
            print("Invalid face region (zero bounding box).")
            config.error_msg = "Invalid face region (zero bounding box)."
            return -1

        symmetry_error = np.linalg.norm(keypoints - flipped_points, axis=1).mean()
        normalized_error = symmetry_error / normalization_factor

        # When normalized_error ~0.26, score ~100
        # As error to 0.36, score approaches 50
    
        score = linear_score(normalized_error)

        print(f"Symmetry error: {symmetry_error:.2f}px | Normalized: {normalized_error:.4f} | Score: {score:.2f}%")

        if draw and frame is not None:
            for (x, y) in keypoints:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
            for (x, y) in flipped_points:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.line(frame, (int(midline), 0), (int(midline), frame.shape[0]), (255, 0, 0), 1)

        return score

    except Exception as e:
        print(f"Error in symmetry score: {e}")
        config.error_msg = f"Error in symmetry score: {e}"
        return -1

def main():
    # capture image from webcam
    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # cap.release()
    # image = frame if ret else None

    # #load image from file
    # # file_path = 'test_faces/smile/1.jpg' 
    image = config.serious_face_image
    frame = image

    config.symmetry_score = get_symmetry_score(image, frame, draw=True)
    # config.face_attractiveness = face_score

    # # Add score text and display
    # cv2.putText(frame, f"Attractiveness: {int(config.emotion_score)}%", (50, 50), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    # cv2.putText(frame, f"symmetry: {int(config.symmetry_score)}%", (50, 50), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    # cv2.imshow("Result", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
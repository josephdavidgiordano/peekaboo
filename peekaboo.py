import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Define function to calculate eye aspect ratio
def get_eye_aspect_ratio(landmarks, eye_indices):
    left = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(left[1] - left[5])
    B = np.linalg.norm(left[2] - left[4])
    C = np.linalg.norm(left[0] - left[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Eye landmarks indices 
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Define threshold for eye aspect ratio
EYE_AR_THRESHOLD = 0.22

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face mesh detection
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in face_landmarks.landmark]

            left_ear = get_eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
            right_ear = get_eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
            ear = (left_ear + right_ear) / 2.0

            # Draw eye contours
            for index in LEFT_EYE_INDICES:
                x, y = landmarks[index]
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            for index in RIGHT_EYE_INDICES:
                x, y = landmarks[index]
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Display "Looking" if detected
            cv2.putText(frame, "Looking", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Display "Not looking" if not detected
        cv2.putText(frame, "Not looking", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display frame with rectangles around faces and eye status
    cv2.imshow('Face and Eye Detection', frame)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

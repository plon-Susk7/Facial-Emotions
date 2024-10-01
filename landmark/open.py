import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Your specific landmark points
landmark_indices =[21, 37, 40, 41, 48, 49, 50, 51, 60, 61, 65, 74, 76, 80, 93, 98, 99, 100, 101, 102, 103, 116, 118, 119, 120, 124, 127, 130, 132, 138, 143, 148, 166, 167, 168, 178, 187, 188, 199, 204, 206, 207, 208, 210, 214, 217, 219, 220, 221, 236, 238, 239, 240, 241, 242, 243, 292, 293, 307, 308, 309, 324, 325, 326, 362, 367, 402, 416, 448, 455]
# Capture video or use an image
cap = cv2.VideoCapture(0)  # Or provide a path to an image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR image to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for idx in landmark_indices:
                h, w, _ = frame.shape

                # Get the position of the landmark
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)

                # Color the landmark point (e.g., red)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Display the image
    cv2.imshow('Face with Colored Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

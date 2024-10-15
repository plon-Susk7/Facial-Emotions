import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Your specific landmark points
landmark_indices = [1,168,205,425,10,190,414,162,389,152]
# Read the image file
image_path = 'iim_nut.jpg'  # Provide the path to your image
image = cv2.imread(image_path)

# Convert BGR image to RGB for processing
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = face_mesh.process(image_rgb)

if result.multi_face_landmarks:
    for face_landmarks in result.multi_face_landmarks:
        for idx in landmark_indices:
            h, w, _ = image.shape

            # Get the position of the landmark
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)

            # Color the landmark point (e.g., red)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# Save the image with colored landmarks
output_path = 'nimstim_single_hpy.jpg'  # Specify the output path
cv2.imwrite(output_path, image)

# Optionally display the image
cv2.imshow('Face with Colored Landmarks', image)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()

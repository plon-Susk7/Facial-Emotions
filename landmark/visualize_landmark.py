import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Path to CSV containing landmark differences
csv_difference_path = 'landmark_differences_csvs/radboud/ANG.csv'
landmark_differences = pd.read_csv(csv_difference_path)

temp = list(landmark_differences.iloc[1,:])

# Let's print max and min values here

# Your specific landmark points
landmark_indices = [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ]
# print(len(landmark_indices))
# Extracting indices whose value is abs(temp[i]) > 20

# landmark_indices = [i for i in range(1,468*2-2)]
# landmark_indices = [i for i in range(len(temp)) if abs(temp[i]) > 20]
# Process landmark indices if needed
def process_landmark_indices(landmark_indices):
    '''
        Function to process landmark indices
    '''
    new_landmark_indices = []
    for i in landmark_indices:
        if i % 2 == 0:
            new_landmark_indices.append(i // 2)
        else:
            new_landmark_indices.append((i-1) // 2 )

    new_landmark_indices = list(set(new_landmark_indices))
    return new_landmark_indices

# landmark_indices = process_landmark_indices(landmark_indices)

print(sorted(landmark_indices))

# Load the image
# image_path = '/home/priyash7/Desktop/Facial-Emotions/landmark/data/radboud/HPY/Rafd090_08_Caucasian_female_happy_frontal.jpg'  # Provide the path to your image
image_path = '../AULP/new_dataset/radboud/HPY/Rafd090_10_Caucasian_male_happy_frontal.jpg'
image = cv2.imread(image_path)

h, w, _ = image.shape

# Convert BGR image to RGB for processing
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = face_mesh.process(image_rgb)

# Set a fixed arrow length
fixed_length = 10 # Change this value to control the arrow size

if result.multi_face_landmarks:
    for face_landmarks in result.multi_face_landmarks:
        h, w, _ = image.shape
        
        for idx in landmark_indices:
            # Get the position of the landmark
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)

            # Overlay the index number on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            color = (0, 255, 0)  # Green color for text
            thickness = 1

            cv2.putText(image, str(idx), (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
# Save the image with standardized arrows
output_path = 'nimstim_single_hpy.jpg'  # Specify the output path
cv2.imwrite(output_path, image)

# Optionally display the image
cv2.imshow('Face with Standardized Arrows',image)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()

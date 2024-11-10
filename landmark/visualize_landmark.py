import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Path to CSV containing landmark differences
csv_difference_path = 'landmark_differences_csvs/radboud/SUR.csv'
landmark_differences = pd.read_csv(csv_difference_path)

# Your specific landmark points
landmark_indices = [30, 32, 34, 36, 38, 66, 88, 156, 168, 170, 172, 174, 176, 178, 180, 182, 184, 192, 194, 214, 272, 274, 278, 282, 294, 298, 300, 302, 306, 340, 342, 344, 346, 352, 354, 358, 360, 362, 364, 366, 390, 400, 402, 404, 406, 410, 418, 422, 424, 526, 628, 630, 632, 634, 636, 638, 640, 642, 644, 672, 730, 732, 736, 740, 756, 758, 760, 790, 792, 794, 796, 802, 806, 808, 810, 812, 814, 838, 844, 846, 850, 858, 862, 864]

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
            new_landmark_indices.append(i // 2 + 1)

    new_landmark_indices = list(set(new_landmark_indices))
    return new_landmark_indices

landmark_indices = process_landmark_indices(landmark_indices)

# Load the image
image_path = 'radboud_nut_sample.jpg'  # Provide the path to your image
image = cv2.imread(image_path)

# Convert BGR image to RGB for processing
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = face_mesh.process(image_rgb)

# Set a fixed arrow length
fixed_length = 20 # Change this value to control the arrow size

if result.multi_face_landmarks:
    for face_landmarks in result.multi_face_landmarks:
        h, w, _ = image.shape
        
        for idx in landmark_indices:
            x_col = idx
            y_col = idx + 1

            # Get the direction from the CSV data
            x_direction = landmark_differences.iloc[:, x_col].mean() * w
            y_direction = landmark_differences.iloc[:, y_col].mean() * h

            # Normalize the direction vector and scale it to the fixed length
            magnitude = np.sqrt(x_direction**2 + y_direction**2)
            if magnitude > 0:
                x_direction = (x_direction / magnitude) * fixed_length
                y_direction = (y_direction / magnitude) * fixed_length

            # Get the position of the landmark
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)

            start_point = (x, y)
            end_point = (int(x + x_direction), int(y + y_direction))
            color = (0, 255, 0)  # Green color for arrows
            thickness = 2

            cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength=0.3)

# Save the image with standardized arrows
output_path = 'nimstim_single_hpy.jpg'  # Specify the output path
cv2.imwrite(output_path, image)

# Optionally display the image
cv2.imshow('Face with Standardized Arrows', image)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()

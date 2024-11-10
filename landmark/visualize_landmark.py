import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Your specific landmark points
# landmark_indices = [1,168,205,425,10,190,414,162,389,152]

##### 1 index

landmark_indices = [38, 66, 88, 118, 168, 214, 272, 274, 278, 282, 298, 300, 302, 306, 340, 342, 344, 346, 352, 354, 366, 390, 400, 402, 404, 406, 410, 418, 422, 424, 430, 432, 526, 578, 628, 672, 730, 732, 736, 740, 756, 758, 760, 790, 792, 794, 796, 802, 814, 838, 844, 846, 850, 858, 862, 864, 872]
def process_landmark_indices(landmark_indices):
    '''
        Function to process landmark indices
    '''
    new_landmark_indices = []
    for i in landmark_indices:
        if i%2==0:
            new_landmark_indices.append(i//2)
        else:
            new_landmark_indices.append(i//2 + 1)

    new_landmark_indices = list(set(new_landmark_indices))
    return new_landmark_indices

landmark_indices = process_landmark_indices(landmark_indices)

image_path = 'radboud_nut_sample.jpg'  # Provide the path to your image
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

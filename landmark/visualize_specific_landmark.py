import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

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

def process_landmark_save_image(landmark_indices):
    # Initialize mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # Path to CSV containing landmark differences
    csv_difference_path = 'landmark_differences_csvs/radboud/SUR.csv'
    landmark_differences = pd.read_csv(csv_difference_path)

    temp = list(landmark_differences.iloc[0,:])

    # Load the image
    
    image_path = 'data/radboud/SUR/Rafd090_01_Caucasian_female_surprised_frontal.jpg'  # Provide the path to your image
    image = cv2.imread(image_path)

    h, w, _ = image.shape

    # Convert BGR image to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(image_rgb)

    # Set a fixed arrow length
    fixed_length = 1 # Change this value to control the arrow size

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = image.shape
            
            for idx in landmark_indices:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    cv2.imwrite('landmark_visualization.jpg',image)



if __name__ == "__main__":
    csv_difference_path = 'landmark_distance_csvs/radboud/SUR.csv'
    landmark_distance_df = pd.read_csv(csv_difference_path)

    landmark_distance_list = list(landmark_distance_df.iloc[0,:])

    print(f"mean : {np.mean(landmark_distance_list)}")
    print(f"max : {np.max(landmark_distance_list)}")
    print(f"min : {np.min(landmark_distance_list)}")
    print(f"std : {np.std(landmark_distance_list)}")
    print(f"median : {np.median(landmark_distance_list)}")
    print(f"len : {len(landmark_distance_list)}")

    # I need to find 70 percentile of the data

    print(f"70 percentile : {np.percentile(landmark_distance_list, 70)}")

    landmark_indices = [i for i in range(len(landmark_distance_list)) if landmark_distance_list[i] > np.percentile(landmark_distance_list, 60)]
    print(len(landmark_indices))
    process_landmark_save_image(landmark_indices)


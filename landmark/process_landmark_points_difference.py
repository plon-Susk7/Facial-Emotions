from helper import *
import os
import numpy as np
import pandas as pd
import argparse


def default_landmark_indices():
    '''
        Function to return default landmark indices in dictionary format

    '''
    # Initialize mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    image_path = './data/radboud/NUT/Rafd090_01_Caucasian_female_neutral_frontal.jpg'
    image = cv2.imread(image_path)

    h,w,_ = image.shape

    # Convert BGR image to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(image_rgb)

    landmark_to_coordinates = {}

    landmark_indices = [i for i in range(0,468)]

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = image.shape
            
            for idx in landmark_indices:
                # Get the position of the landmark
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)

                landmark_to_coordinates[idx] = [x,y]

    return landmark_to_coordinates

def getLandmarkPoint(dataset):
    '''
        Function to get displacemnet of landmark points in given dataset
        
        Args:
            dataset : Path to the dataset

        Returns:
            Nothing.
            Saves the csv file in the landmark_distance_csvs folder
    '''
    print(f"Processing {dataset}")
    first_path = f'./data/{dataset}/NUT' #NUT
    # first_path = f'../AULP/new_dataset/{dataset}/NUT' 
    
    first_dataset = os.listdir(first_path)

    emotions=["HPY","FER","ANG","SAD","SUR"]

    for emotion in emotions:
        second_path = f"./data/{dataset}/{emotion}"
        # second_path = f'../AULP/new_dataset/{dataset}/{emotion}'
        
        second_dataset = os.listdir(second_path)
    
        result = []

        # We need to create pairs somehow, pairs of path 
        # We'll create pairs of images from the two datasets
        for path in first_dataset:
            
            for path2 in second_dataset:

                
                # if f"{path.split('_')[0]}_{path.split('_')[1]}" == f"{path2.split('_')[0]}_{path2.split('_')[1]}":
                #     result.append((os.path.join(first_path,path),os.path.join(second_path,path2)))

                # FOr jAFFE
                if f"{path.split('_')[0]}_" == f"{path2.split('_')[0]}_":
                    result.append((os.path.join(first_path,path),os.path.join(second_path,path2)))

            if(len(result)==len(second_dataset)):
                break

        
        final = []
        landmark_to_coordinates = default_landmark_indices()
        dst_matrix = [landmark_to_coordinates[x] for x in landmark_to_coordinates]
        for pair in result:
            temp = [] # We'll store the displacement here!
            pointsA = getLandmarkPoints(pair[0])
            _,b1 = affine_trans(np.array(pointsA),dst_matrix)
            # _,b1 = similarity_trans(points,landmark_to_coordinates)

            pointsB = getLandmarkPoints(pair[1])
            _,b2 = affine_trans(np.array(pointsB),dst_matrix)
            # _,b2 = similarity_trans(points,landmark_to_coordinates)

            for i in range(len(b1)):
                temp.append(b1[i][0]-b2[i][0])
                temp.append(b1[i][1]-b2[i][1])
            
            final.append(temp)
        
        
        df = pd.DataFrame(columns=range(0,469*2-2),data=final)

        if os.path.exists(f"./landmark_differences_csvs/{dataset}") == False:
            os.makedirs(f"./landmark_differences_csvs/{dataset}")
        path_to_save=f"./landmark_differences_csvs/{dataset}/{emotion}.csv"
        
        df.to_csv(path_to_save,index=False)
        print("Saved!")


if __name__ == "__main__":

    # python index.py <dataset_name>
    parser = argparse.ArgumentParser(description="Process landmark points distance")
    parser.add_argument("dataset",type=str,help="Dataset to process")

    args = parser.parse_args()
    getLandmarkPoint(args.dataset)
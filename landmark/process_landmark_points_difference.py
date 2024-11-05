from helper import *
import os
import numpy as np
import pandas as pd
import argparse

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
    
    first_dataset = os.listdir(first_path)

    emotions=["ANG","FER","HPY","SAD","SUR"]

    for emotion in emotions:
        second_path = f"./data/{dataset}/{emotion}"
        
        second_dataset = os.listdir(second_path)
    
        result = []

        # We need to create pairs somehow, pairs of path 
        # We'll create pairs of images from the two datasets
        for path in first_dataset:
            
            for path2 in second_dataset:

                
                if f"{path.split('_')[0]}_{path.split('_')[1]}" == f"{path2.split('_')[0]}_{path2.split('_')[1]}":
                    result.append((os.path.join(first_path,path),os.path.join(second_path,path2)))

            if(len(result)==len(second_dataset)):
                break

        final = []


        for pair in result:
            temp = [] # We'll store the displacement here!
            pointsA = getLandmarkPoints(pair[0])
            _,points = affine_trans(np.array(pointsA))
            _,b1 = similarity_trans(points)

            pointsB = getLandmarkPoints(pair[1])
            _,points = affine_trans(np.array(pointsB))
            _,b2 = similarity_trans(points)

            for i in range(len(b1)):
                # temp.append(((b1[i][0]-b2[i][0])**2 + (b1[i][1]-b2[i][1])**2)**0.5)
                temp.append(b1[i][0]-b2[i][0])
                temp.append(b1[i][1]-b2[i][1])
            
            final.append(temp)
        
        
        df = pd.DataFrame(columns=range(1,469*2-1),data=final)
        path_to_save=f"./landmark_differences_csvs/{dataset}/{emotion}.csv"
        df.to_csv(path_to_save,index=False)


if __name__ == "__main__":

    # python index.py <dataset_name>
    parser = argparse.ArgumentParser(description="Process landmark points distance")
    parser.add_argument("dataset",type=str,help="Dataset to process")

    args = parser.parse_args()
    getLandmarkPoint(args.dataset)
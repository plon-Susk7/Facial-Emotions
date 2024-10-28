from helper import *
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':

    first_path = './data/radboud/NUT' #NUT

    first_dataset = os.listdir(first_path)
    emotions=["ANG","FER","HPY","SAD","SUR"]

    for emotion in emotions:
        second_path = "./data/radboud/"+emotion
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
                temp.append(((b1[i][0]-b2[i][0])**2 + (b1[i][1]-b2[i][1])**2)**0.5)
            
            final.append(temp)
        
        
        df = pd.DataFrame(columns=range(1,469),data=final)
        path_to_save=f"./landmark_distance_csvs/radboud/{emotion}.csv"
        df.to_csv(path_to_save,index=False)
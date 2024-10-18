from helper import *
import os
import numpy as np
import pandas as pd

'''
    # 162 -> 300,900
    # 190 -> 600,1000
    414 -> 800,1000
    # 168 -> 700,900
    # 10 -> 700, 1300
    # 1 -> 700, 600
    205 -> 500, 600
    425 -> 900, 600
    # 152 -> 500. 100
'''


if __name__ == '__main__':

    first_path = './data/iim/NUT' #NUT

    first_dataset = os.listdir(first_path)
    emotions=["ANG","FER","HPY","SAD","SUR"]

    for emotion in emotions:
        second_path = "./data/iim/"+emotion
        second_dataset = os.listdir(second_path)
        result = []

        # We need to create pairs somehow, pairs of path 
        # We'll create pairs of images from the two datasets
        for path in first_dataset:
            for path2 in second_dataset:
                if path.split('.')[0][0:-3] == path2.split('.')[0][0:-3]:
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
        path_to_save=f"./landmark_distance_csvs/iim/{emotion}.csv"
        df.to_csv(path_to_save,index=False)
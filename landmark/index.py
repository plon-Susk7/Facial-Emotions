# from test import getLandmarkPoints
from helper import *
import os
import pandas as pd
import numpy as np



if __name__ == "__main__":

    """ We need two folder paths, both of these folder paths need to be from same dataset."""

    first_path = 'test_data/A' #NUT
    second_path = 'test_data/B' #HPY

    # we need to get image pairs from these two folders
    # we'll differentiate between images with the prefix, for eg 0_Neutral.jpg and 0_happy.jpg

    first_dataset = os.listdir(first_path)
    second_dataset = os.listdir(second_path)

    # We need to create pairs somehow, pairs of path 
    # We'll create pairs of images from the two datasets
    result = []

    for path in first_dataset:
        for path2 in second_dataset:
            if path.split('_')[0] == path2.split('_')[0]:
                result.append((os.path.join(first_path,path),os.path.join(second_path,path2)))
        
        if(len(result)==len(second_dataset)):
            break

    final = []
    for pair in result:
        temp = [] # We'll store the displacement here!
        pointsA = getLandmarkPoints(pair[0])
        pointsB = getLandmarkPoints(pair[1])

        for i in range(len(pointsA)):
            temp.append(((pointsA[i][0]-pointsB[i][0])**2 + (pointsA[i][1]-pointsB[i][1])**2 + (pointsA[i][2]-pointsB[i][2])**2)**0.5)
        
        final.append(temp)

    df = pd.DataFrame(columns=range(1,469),data=final)
    df.to_csv('results.csv',index=False)

    
        
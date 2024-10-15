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

def affine_trans(kp, data_name=None, matrix=None):
    src = np.float32([kp[1], kp[10], kp[152], kp[162], kp[168], kp[190], kp[205], kp[425], kp[414]])
    dst = np.float32([[700,600], [700, 1300], [500, 100], [300, 900],[700,900],[600,1000],[500,600],[900,600],[800,1000]])
    
    src_h = np.concatenate((src.T, np.ones(src.shape[0]).reshape(1,src.shape[0])), axis=0)
    dst_h = np.concatenate((dst.T, np.ones(dst.shape[0]).reshape(1,dst.shape[0])), axis=0)
    
    kp_h = np.concatenate((kp.T, np.ones(kp.shape[0]).reshape(1,kp.shape[0])), axis=0)
    
    if matrix is None: matrix = dst_h@np.linalg.pinv(src_h)
    kp_h = matrix@kp_h
    
    return matrix, kp_h[:2, :].T


if __name__ == '__main__':

    first_path = 'test_data/A' #NUT
    second_path = 'test_data/B' #HPY

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

        a1,b1 = affine_trans(np.array(pointsA))
        pointsB = getLandmarkPoints(pair[1])
        a2,b2 = affine_trans(np.array(pointsB))

        for i in range(len(b1)):
            temp.append(((b1[i][0]-b2[i][0])**2 + (b1[i][1]-b2[i][1])**2)**0.5)
        
        final.append(temp)
    df = pd.DataFrame(columns=range(1,469),data=final)
    df.to_csv('results.csv',index=False)
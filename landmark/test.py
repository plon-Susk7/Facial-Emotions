from helper import *
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

    file_path = 'test_data/A/0_NUT.png' #NUT

    points = getLandmarkPoints(file_path)

    a,b = affine_trans(np.array(points))
    _,b = similarity_trans(b)
    # Now use matplotlib to plot b, b is 2d vector with x and y coordinates and save image

    plt.scatter(b[:,0],b[:,1])
    plt.savefig('test_data/A/0_NUT_landmarks.png')



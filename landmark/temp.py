from helper import *
import os
import numpy as np
import pandas as pd
import argparse

from matplotlib import pyplot as plt

nut_path = '../AULP/new_dataset/radboud/NUT/Rafd090_01_Caucasian_female_neutral_frontal.jpg'


pointsA = getLandmarkPoints(nut_path)
_,points = affine_trans(np.array(pointsA))
_,b1 = similarity_trans(points)

# Now we plot the points

plt.scatter(points[:,0],points[:,1])
plt.show()
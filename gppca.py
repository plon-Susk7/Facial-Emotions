from helper import *
import pandas as pd
import numpy as np
from statsmodels.multivariate.factor_rotation import rotate_factors
import os

threshold=60
varianceExplained=0.95
emotion = "ANG"

jaffe= pd.read_csv("./landmark/landmark_distance_csvs/jaffe/SUR.csv")
iim= pd.read_csv("./landmark/landmark_distance_csvs/iim/SUR.csv")
radboud = pd.read_csv("./landmark/landmark_distance_csvs/radboud/SUR.csv")

min_val = min(jaffe.shape[0],iim.shape[0])

jaffe_numpy = jaffe.iloc[:min_val,:].to_numpy()
iim_numpy = iim.iloc[:min_val,:].to_numpy()
radboud_numpy = radboud.iloc[:min_val,:].to_numpy()


allData = {"iim":iim_numpy,"jaffe":jaffe_numpy,"radboud":radboud_numpy}

w,v = GeneralizedPPCA(allData)

loadings = getLoadings(w,v,varianceExplained)
rotated_loadings, _ = rotate_factors(loadings.T, 'varimax')

# print(rotated_loadings.shape)
plotHeatMap(rotated_loadings,iim.columns[:-1],emotion,'all','all')
getRowsWithExtremeValues(rotated_loadings, iim.columns, threshold)
plotHeatMapWithRowAnnotations(rotated_loadings, iim.columns, emotion, 'single', 'all', threshold)


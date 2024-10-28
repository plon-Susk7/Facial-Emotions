from helper import *
import pandas as pd
import numpy as np
from statsmodels.multivariate.factor_rotation import rotate_factors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def main():
    iim_path = 'landmark/landmark_distance_csvs/radboud/'

    emotions = ["HPY", "ANG", "SAD", "FER", "SUR"]
    threshold = 0.065
    variance_explained = 0.95

    for emotion in emotions:
        extreme_rows_per_dataset = {}
        iim_path_emotion = f"{iim_path}{emotion}.csv"
        df1 = pd.read_csv(iim_path_emotion)
        
        # scaler = StandardScaler()
        # df1_scaled = scaler.fit_transform(df1)
        # Perform PCA
        pca = PCA(n_components=variance_explained)
        pca.fit(df1)
        
        w = pca.components_.T  # Transpose to get features x components
        
        v = pca.explained_variance_ratio_ 
        print(w)
        loadings = getLoadings(w,v,variance_explained)
        # Get loadings (components) and explained variance
        # loadings = pca.components_.T
        
        # Rotate loadings using Varimax
        rotated_loadings, _ = rotate_factors(loadings.T, 'varimax')
        
        # Plot heatmap of rotated loadings
        plotHeatMap(rotated_loadings, df1.columns, emotion, 'single', 'iim')
        
        # Identify rows with extreme values
        extreme_rows = getRowsWithExtremeValues(rotated_loadings, df1.columns, threshold)
        extreme_rows_per_dataset['iim'] = set(extreme_rows)
        
        # Plot heatmap with row annotations
        plotHeatMapWithRowAnnotations(rotated_loadings, df1.columns, emotion, 'single', 'iim', threshold)

if __name__ == "__main__":
    main()

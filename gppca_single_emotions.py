# from helper import *
# import pandas as pd
# import numpy as np
# from statsmodels.multivariate.factor_rotation import rotate_factors
# import os

# def main():
#     iim_path = 'landmark/landmark_distance_csvs/iim/'

#     emotions = ["HPY","ANG","SAD","FER","SUR"]
#     threshold=15
#     varianceExplained=0.95

#     for emotion in emotions:
#         extreme_rows_per_dataset={}
#         iim_path_emotion = f"{iim_path}{emotion}.csv"
#         df1 = pd.read_csv(iim_path_emotion)
#         rows,cols = df1.shape
#         df2 = df1.sample(n=rows,random_state=42)

#         gppcaFeader = {'data1' : df1.to_numpy(),'data2' : df2.to_numpy()}

#         w,v = GeneralizedPPCA(gppcaFeader)
#         loadings = getLoadings(w,v,varianceExplained)
#         rotated_loadings, _ = rotate_factors(loadings.T, 'varimax')
#         plotHeatMap(rotated_loadings,df1.columns,emotion,'single','iim')
#         extreme_rows = getRowsWithExtremeValues(rotated_loadings, df1.columns, threshold)
#         extreme_rows_per_dataset['iim'] = set(extreme_rows)

#         plotHeatMapWithRowAnnotations(rotated_loadings, df1.columns, emotion, 'single', 'iim', threshold)



# # def main():

# #     iim_path = 'landmark/landmark_distance_csvs/iim/'

# #     # Loading all the data in pandas dataframes
# #     # ckplus = pd.read_csv("au_dataset/results_ckplus.csv")
# #     # iim = pd.read_csv("landmark/landmark_distance_csvs/iim/")
# #     # jaffe = pd.read_csv("landmark_dataset/results_jaffe.csv")
# #     # nimstim = pd.read_csv("landmark_dataset/results_nimstim.csv")
# #     # radboud = pd.read_csv("au_dataset/results_radboud.csv")

# #     # Preprocessing the data
# #     emotions = ["HPY","ANG","SAD","FER","SUR"]
# #     allData = {"iim":iim} ### Need to parameter to fiddle this, maybe add or delete few datasets
# #     threshold=0.5
# #     varianceExplained=0.95
# #     for emotion in emotions:
# #         extreme_rows_per_dataset={}
# #         for key in allData.keys():
# #             print(f"Emotion: {emotion} Dataset: {key}")
# #             df = allData[key]
# #             filtered_df1 = df[df[df.columns[-1]].str.endswith(f'{emotion}.csv')].iloc[:, :-1]

# #             rows,cols = filtered_df1.shape
# #             filtered_df2 = filtered_df1.sample(n=rows,random_state=42)

# #             gppcaFeader = {'data1' : filtered_df1.to_numpy(),'data2' : filtered_df2.to_numpy()}
# #             w,v = GeneralizedPPCA(gppcaFeader)
# #             loadings = getLoadings(w,v,varianceExplained) # getLoadings(w,v,variance_explained)
# #             rotated_loadings, _ = rotate_factors(loadings.T, 'varimax')
# #             plotHeatMap(rotated_loadings,iim.columns[:-1],emotion,'single',key)
# #             extreme_rows = getRowsWithExtremeValues(rotated_loadings, iim.columns[:-1], threshold)
# #             extreme_rows_per_dataset[key] = set(extreme_rows)

# #             plotHeatMapWithRowAnnotations(rotated_loadings, iim.columns[:-1], emotion, 'single', key, threshold)

# #         # Find common extreme rows across all datasets for this emotion
# #         common_extreme_rows = set.intersection(*extreme_rows_per_dataset.values())
# #         print(f"Common extreme rows for emotion {emotion}: {common_extreme_rows}")


# if __name__ == "__main__":
#     main()

from helper import *
import pandas as pd
import numpy as np
from statsmodels.multivariate.factor_rotation import rotate_factors
from sklearn.decomposition import PCA
import os

def main():
    iim_path = 'landmark/landmark_distance_csvs/iim/'

    emotions = ["HPY", "ANG", "SAD", "FER", "SUR"]
    threshold = 0.08
    variance_explained = 0.95

    for emotion in emotions:
        extreme_rows_per_dataset = {}
        iim_path_emotion = f"{iim_path}{emotion}.csv"
        df1 = pd.read_csv(iim_path_emotion)
        
        # Perform PCA
        pca = PCA(n_components=variance_explained)
        pca.fit(df1)
        
        w = pca.components_.T  # Transpose to get features x components
        v = pca.explained_variance_ratio_ 
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

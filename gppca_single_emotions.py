from helper import *
import pandas as pd
import numpy as np
from statsmodels.multivariate.factor_rotation import rotate_factors
import os


def main():
    # Loading all the data in pandas dataframes
    ckplus = pd.read_csv("au_dataset/results_ckplus.csv")
    iim = pd.read_csv("au_dataset/results_iim.csv")
    jaffe = pd.read_csv("au_dataset/results_jaffe.csv")
    nimstim = pd.read_csv("au_dataset/results_nimstim.csv")
    radboud = pd.read_csv("au_dataset/results_radboud.csv")

    # Preprocessing the data
    emotions = ["ANG","SAD","FER","SUR",'HPY']
    allData = {"ckplus":ckplus,"iim":iim,"jaffe":jaffe,"nimstim":nimstim,"radboud":radboud} ### Need to parameter to fiddle this, maybe add or delete few datasets
    threshold=0.25
    varianceExplained=0.95
    for emotion in emotions:
        extreme_rows_per_dataset={}
        for key in allData.keys():
            print(f"Emotion: {emotion} Dataset: {key}")
            df = allData[key]
            filtered_df1 = df[df[df.columns[-1]].str.endswith(f'{emotion}.csv')].iloc[:, :-1]

            rows,cols = filtered_df1.shape
            filtered_df2 = filtered_df1.sample(n=rows,random_state=42)

            gppcaFeader = {'data1' : filtered_df1.to_numpy(),'data2' : filtered_df2.to_numpy()}
            w,v = GeneralizedPPCA(gppcaFeader)
            loadings = getLoadings(w,v,varianceExplained) # getLoadings(w,v,variance_explained)
            rotated_loadings, _ = rotate_factors(loadings.T, 'varimax')
            plotHeatMap(rotated_loadings,ckplus.columns[:-1],emotion,'single',key)
            extreme_rows = getRowsWithExtremeValues(rotated_loadings, ckplus.columns[:-1], threshold)
            extreme_rows_per_dataset[key] = set(extreme_rows)

            plotHeatMapWithRowAnnotations(rotated_loadings, ckplus.columns[:-1], emotion, 'single', key, threshold)

        # Find common extreme rows across all datasets for this emotion
        common_extreme_rows = set.intersection(*extreme_rows_per_dataset.values())
        print(f"Common extreme rows for emotion {emotion}: {common_extreme_rows}")


if __name__ == "__main__":
    main()
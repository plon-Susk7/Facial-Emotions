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
    emotions = ["ANG","SAD","FER","SUR"]
    allData = {"ckplus":ckplus,"iim":iim,"jaffe":jaffe,"nimstim":nimstim,"radboud":radboud} ### Need to parameter to fiddle this, maybe add or delete few datasets

    for emotion in emotions:
        for key in allData.keys():
            print(f"Emotion: {emotion} Dataset: {key}")
            df = allData[key]
            filtered_df1 = df[df[df.columns[-1]].str.endswith(f'{emotion}.csv')].iloc[:, :-1]

            rows,cols = filtered_df1.shape
            filtered_df2 = filtered_df1.sample(n=rows,random_state=42)

            gppcaFeader = {'data1' : filtered_df1.to_numpy(),'data2' : filtered_df2.to_numpy()}
            w,v = GeneralizedPPCA(gppcaFeader)
            loadings = getLoadings(w,v)
            rotated_loadings, _ = rotate_factors(loadings.T, 'varimax')
            plotHeatMap(rotated_loadings,ckplus.columns[:-1],emotion,'single',key)


if __name__ == "__main__":
    main()
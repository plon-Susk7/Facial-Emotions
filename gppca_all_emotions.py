from helper import *
import pandas as pd
import numpy as np
from statsmodels.multivariate.factor_rotation import rotate_factors
import os











def preprocess_data(data,emotions):
    '''
    Function to preprocess the data as per the emotions
    Input:
        data : Dictionary of dataframes
        emotions : List of emotions to preprocess the data [ANG,SAD,FER,SUR]
    '''
    preprocessed_data = {'ANG':{},'SAD':{},'FER':{},'SUR':{}}
    ## Preprocessing the data as per given emotion [ANG,SAD,FER,SUR]
    for emotion in emotions:
        
        minimum_sample = min([len([x for x in data[key]['filename'].str.endswith(f'{emotion}.csv') if x==True]) for key in data.keys()])
        print(f"Minimum Sample for {emotion} is {minimum_sample}")

        for key in data.keys():
            df = data[key]
            filtered_df = df[df[df.columns[-1]].str.endswith(f'{emotion}.csv')]
            # print(filtered_df2.shape)
            filtered_df = filtered_df.sample(n=minimum_sample,random_state=42)
            filtered_df = filtered_df.iloc[:, :-1]
            dataPoints = filtered_df.to_numpy()
            preprocessed_data[emotion][key] = dataPoints

    return preprocessed_data






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
    # print([x for x in allData["ckplus"]['filename'].str.endswith("ANG.csv") if x==True])

    preprocessedData = preprocess_data(allData,emotions)

    # Applying Generalized PPCA and plot data
    for emotion in emotions:
        print(f"Emotion: {emotion}")
        all_data = preprocessedData[emotion]

        print(all_data['ckplus'].shape)
        w,v = GeneralizedPPCA(all_data)
        loadings = getLoadings(w,v)
        rotated_loadings, _ = rotate_factors(loadings.T, 'varimax')
    
        print(rotated_loadings.shape)
        plotHeatMap(rotated_loadings,ckplus.columns[:-1],emotion,'all')

        



if __name__ == "__main__":
    main()
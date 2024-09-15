import itertools
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.multivariate.factor_rotation import rotate_factors
from scipy.stats import norm
import os

def GeneralizedPPCA(all_data):
    """
    Input: dict of key:data (ndarray) pairs

    Returns PPCA weights across all data matrices stored in the input dict
    """
    action_units = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r',
       ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r',
       ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']
    for key in all_data.keys():
        d = all_data[key]
        assert np.isnan(d).sum() == 0
        d = d - d.mean(axis=0) 
        all_data[key] = d

    pairwise_combinations = list(itertools.combinations(all_data.keys(), 2))
    
#     print(pairwise_combinations)
    if not pairwise_combinations:
        # Handle the case with only one dataset
        key = next(iter(all_data.keys()))
        a = all_data[key]
        crosscov = np.matmul(a.transpose(), a)
        print("here")

    else:
        for key1, key2 in pairwise_combinations:
            a = all_data[key1]
            b = all_data[key2]
            
            if (key1, key2) == pairwise_combinations[0]:
                crosscov = ( np.matmul(a.transpose(), b) + np.matmul(b.transpose(), a))
            else:
                crosscov += ( np.matmul(a.transpose(), b) + np.matmul(b.transpose(), a))
    
# #     crosscov/=len(all_data)
#     plt.figure(figsize=(12, 8))  # Adjust the width and height as needed

# # Create the heatmap with annotations
#     sns.heatmap(crosscov, annot=True, fmt=".2f", annot_kws={"size": 10},xticklabels=action_units,yticklabels=action_units)  # Adjust font size with 'annot_kws'

#     plt.show()
    # v is the eigenvalues (or component covariances)
    # w is the eigenvectors (or PPCs)
    v, w = LA.eigh(crosscov)
    w = np.flip(w, 1)  # reverse w so it is in descending order of eigenvalue
    v = np.flip(v)  # reverse v so it is in descending order
    return w, v

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):

    ''' 
    Rotates the matrix Phi using the varimax algorithm. May not be of use
    since we're using the statsmodels implementation of varimax rotation.
    '''
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)

def getLoadings(w,v):
    '''
        Takes two arguments eigenvectors and eigenvalues and returns the loadings
    '''
    return np.array([np.sqrt(val) * w[:, i] for i, val in enumerate(v) if val > 0])

def plotHeatMap(rotated_loadings, columns,emotion,dataset_flag,dataset):
    '''
        Plots the heatmap of the rotated loadings.
        Takes two arguments rotated loadings and the filtered dataframe
    '''
    results_dir = 'results'
    if dataset not in os.listdir(results_dir):
        os.mkdir(f'results/{dataset}')

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        rotated_loadings, 
        annot=True, 
        fmt=".2f", 
        cmap='viridis', 
        cbar=True, 
        linecolor='gray', 
        yticklabels=columns
    )
    plt.title('Heatmap from Rotated Loadings')
    plt.savefig(f'results/{dataset}/heatmap_{emotion}_{dataset_flag}.png')
    # plt.show()
    

def plotCumVar(eig_vals):
    '''
        Plots the cumulative variance explained by the principal components
        Takes one argument eig_vals
    '''

    eig_vals = [x for x in eig_vals if x > 0]
    n = len(eig_vals)
    total = sum(eig_vals)
    var_exp = [ (i/total)*100  for i in sorted(eig_vals,reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    print('Variance Explained: ',var_exp)
    print('Cummulative Variance Explained: ',cum_var_exp)
    plt.bar(range(n),var_exp, align='center',color='lightgreen',edgecolor='black',label='Indiviual Explained Varinace')
    plt.step(range(n), cum_var_exp, where='mid',color='red',label='Cummulative explained Variance')
    plt.legend(loc = 'best')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.tight_layout()
    plt.show()
    
def plotGaussianCurve(data_map, au):

    '''
        Plots the Gaussian curve for the given data
        Takes two arguments data_map and au
    '''
    mean_std_map = {}
    pdf_map = {}
    xmin = 0
    xmax = 0
    
    # Define a list of colors
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Add more if needed
    
    for data in data_map:
        mu, std = norm.fit(data_map[data])
        mean_std_map[data] = [mu, std]
        xmin = min(xmin, min(data_map[data])) 
        xmax = max(xmax, max(data_map[data]))
    
    x = np.linspace(xmin, xmax, 100)
    
    for data in mean_std_map:
        mu = mean_std_map[data][0]
        std = mean_std_map[data][1]
        pdf_map[data] = norm.pdf(x, mu, std)
    
    # Plot each Gaussian curve with a different color
    for i, data in enumerate(pdf_map):
        pdf = pdf_map[data]
        mu = mean_std_map[data][0]
        std = mean_std_map[data][1]
        
        # Cycle through colors, use modulo to handle more datasets than colors
        plt.plot(x, pdf, color=colors[i % len(colors)], label=f'{data}: μ={mu:.2f}, σ={std:.2f}')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Gaussian Curves for {au}')
    plt.legend()

    # Display the plot
    plt.show()
    
        
    
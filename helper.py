import itertools
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.multivariate.factor_rotation import rotate_factors
from scipy.stats import norm
import os
import cv2
import mediapipe as mp
from skimage.transform import SimilarityTransform


def GeneralizedPPCA(all_data):
    """
    Input: dict of key:data (ndarray) pairs

    Returns PPCA weights across all data matrices stored in the input dict
    """
    # action_units = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r',
    #    ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r',
    #    ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']

    action_units = [i for i in range(0,469)]
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

def getLoadings(w,v,varianceExplained=0.95):
    '''
        Takes two arguments eigenvectors, eigenvalues and varianceExplained to returns the loadings
    '''
    eig_vals = [x for x in v if x > 0]
    n = len(eig_vals)
    total = sum(eig_vals)
    var_exp = [ (i/total)*100  for i in sorted(eig_vals,reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    maxIndex = np.searchsorted(cum_var_exp, varianceExplained * 100, side='right')
    loadings =  np.array([np.sqrt(val) * w[:, i] for i, val in enumerate(v) if val > 0])
    return loadings[:maxIndex+1]



def plotHeatMap(rotated_loadings, columns,emotion,dataset_flag,dataset):
    '''
        Plots the heatmap of the rotated loadings.
        Takes two arguments rotated loadings and the filtered dataframe
    '''
    results_dir = 'results_landmark'
    if dataset not in os.listdir(results_dir):
        os.mkdir(f'results_landmark/{dataset}')

    plt.figure(figsize=(100, 80))
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
    plt.savefig(f'results_landmark/{dataset}/heatmap_{emotion}_{dataset_flag}.png')
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
    
def getRowsWithExtremeValues(rotated_loadings, columns, threshold=0.5):
    extreme_rows = []
    print("*"*10)
    for i, row in enumerate(rotated_loadings):
        extreme_values = [(val, idx) for idx, val in enumerate(row) if val > threshold or val < -threshold]
        
        if extreme_values:

            # print(f"{columns[i]},",end="")
            extreme_rows.append(columns[i])
        
        
    print([int(x) for x in extreme_rows])
    print("*"*10)
    return extreme_rows

def plotHeatMapWithRowAnnotations(rotated_loadings, columns, emotion, dataset_flag, dataset, threshold=0.5):
    '''
    Plots the heatmap of the rotated loadings and annotates rows with values greater than the threshold or less than -threshold.
    
    Parameters:
    - rotated_loadings: The matrix of rotated loadings (numpy array).
    - columns: The row names (typically the Action Units).
    - emotion: The emotion for which the heatmap is plotted.
    - dataset_flag: Indicates the type of dataset for labeling.
    - dataset: The dataset name (used for saving the plot).
    - threshold: The threshold to consider for extreme values (default is 0.5).
    '''
    results_dir = 'results_landmark'
    if dataset not in os.listdir(results_dir):
        os.mkdir(f'results_landmark/{dataset}')

    # Create the heatmap
    plt.figure(figsize=(100, 80))  # Adjusted size to make room for text annotations
    sns.heatmap(
        rotated_loadings, 
        annot=True, 
        fmt=".2f", 
        cmap='viridis', 
        cbar=True, 
        linecolor='gray', 
        yticklabels=columns
    )
    
    # Find rows that have values greater than the threshold or less than -threshold
    annotated_rows = []
    for i, row in enumerate(rotated_loadings):
        if np.any(row > threshold) or np.any(row < -threshold):
            annotated_rows.append(columns[i])
    
    # Add the annotated rows to the right of the heatmap
    plt.text(len(rotated_loadings[0]) + 1, 0.5,  # Position the text to the right of the heatmap
             '\n'.join(annotated_rows),  # List rows from top to bottom
             fontsize=12, va='top', ha='left')

    plt.title(f'Heatmap with Row Annotations ({emotion})')
    
    # Save the plot with the annotated rows
    plt.savefig(f'results_landmark/{dataset}/heatmap_{emotion}_{dataset_flag}_annotated.png', bbox_inches='tight')
    # plt.show()

def plotLandmarkIndicesToImage(landmark_indices,image_path,output_path):
    import cv2
    import mediapipe as mp
    import numpy as np

    # Initialize mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()


    # image_path = 'radboud_nut.jpg'  # Provide the path to your image
    image = cv2.imread(image_path)

    # Convert BGR image to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(image_rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for idx in landmark_indices:
                h, w, _ = image.shape

                # Get the position of the landmark
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)

                # Color the landmark point (e.g., red)
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    # Save the image with colored landmarks
    # output_path = 'nimstim_single_hpy.jpg'  # Specify the output path
    cv2.imwrite(output_path, image)

    # Optionally display the image
    cv2.imshow('Face with Colored Landmarks', image)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()

def similarity_trans(kp,matrix=None):

    # Registering right eyebrows and eyes
    src = kp[[130,133]]
    dst = np.array([[400,900],[600,900]])
    tform = SimilarityTransform()
    tform.estimate(src, dst)

    # Keypoints related to right eyebrow and eyes -> https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
    right_eye_kps = [246, 161, 160, 159, 158, 157, 173,33, 7, 163, 144, 145, 153, 154, 155, 133,247, 30, 29, 27, 28, 56, 190,130, 25, 110, 24, 23, 22, 26, 112, 243,113, 225, 224, 223, 222, 221, 189,226, 31, 228, 229, 230, 231, 232, 233, 244,143, 111, 117, 118, 119, 120, 121, 128, 245]
    kp[right_eye_kps] = tform(kp[right_eye_kps])


    # Registering left eyebrows and eyes
    src = kp[[463, 359]]
    dst = np.array([[800,900],[1000,900]])
    tform = SimilarityTransform()
    tform.estimate(src, dst)
    left_eye_kps = [466, 388, 387, 386, 385, 384, 398,263, 249, 390, 373, 374, 380, 381, 382, 362,467, 260, 259, 257, 258, 286, 414,359, 255, 339, 254, 253, 252, 256, 341, 463,342, 445, 444, 443, 442, 441, 413,446, 261, 448, 449, 450, 451, 452, 453, 464,372, 340, 346, 347, 348, 349, 350, 357, 465]
    kp[left_eye_kps] = tform(kp[left_eye_kps])

    # Registering jawline
    src = kp[[162, 389]]
    dst = np.array([[300,1000],[1100,1000]])
    tform = SimilarityTransform()
    tform.estimate(src, dst)
    jawline_kps = [
    10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
    ]
    kp[jawline_kps] = tform(kp[jawline_kps])

    return np.array(matrix), kp



def affine_trans(kp, data_name=None, matrix=None):

    '''
    # 162 -> 300,1000
    # 190 -> 600,900
    414 -> 800,1000
    # 168 -> 700,900
    # 10 -> 700, 1300
    # 1 -> 700, 600
    205 -> 500, 600
    425 -> 900, 600
    # 152 -> 500. 100
    # 389 -> 1000,1000
'''

    src = np.float32([kp[162], kp[389], kp[6], kp[1], kp[133], kp[463]])
    dst = np.float32([[300,1000], [1100, 1000], [700, 900], [700, 500],[600,900],[800,900]])
    
    src_h = np.concatenate((src.T, np.ones(src.shape[0]).reshape(1,src.shape[0])), axis=0)
    dst_h = np.concatenate((dst.T, np.ones(dst.shape[0]).reshape(1,dst.shape[0])), axis=0)
    
    kp_h = np.concatenate((kp.T, np.ones(kp.shape[0]).reshape(1,kp.shape[0])), axis=0)
    
    if matrix is None: matrix = dst_h@np.linalg.pinv(src_h)
    kp_h = matrix@kp_h
    
    return matrix, kp_h[:2, :].T

def getLandmarkPoints(image_path):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    results = []
    # Load image
    image = cv2.imread(image_path)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get the landmarks
    result = face_mesh.process(image_rgb)

    # Check if landmarks were detected
    if result.multi_face_landmarks:
        
        for face_landmarks in result.multi_face_landmarks:
            
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Print the landmark index and its coordinates (x, y, z)
                
                results.append([landmark.x,landmark.y])
    else:
        print("No landmarks detected.")

    # Release the FaceMesh resources
    face_mesh.close()
    return results
    
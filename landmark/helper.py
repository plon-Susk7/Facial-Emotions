import cv2
import mediapipe as mp
import os
import numpy as np
from skimage.transform import SimilarityTransform


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
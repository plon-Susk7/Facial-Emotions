import cv2
import mediapipe as mp
import os
import numpy as np

def affine_trans(kp, data_name=None, matrix=None):

    '''
    # 162 -> 300,900
    # 190 -> 600,1000
    414 -> 800,1000
    # 168 -> 700,900
    # 10 -> 700, 1300
    # 1 -> 700, 600
    205 -> 500, 600
    425 -> 900, 600
    # 152 -> 500. 100
'''

    src = np.float32([kp[1], kp[10], kp[152], kp[162], kp[168], kp[190], kp[205], kp[425], kp[414]])
    dst = np.float32([[700,600], [700, 1300], [500, 100], [300, 900],[700,900],[600,1000],[500,600],[900,600],[800,1000]])
    
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
import cv2
import mediapipe as mp
import os



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
import os
import cv2
import mediapipe as mp


def crop_images(image_path, save_path):
    # Load Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection

    # Initialize the face detection model
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    height, width, _ = image.shape

    # Convert to RGB (Mediapipe requires RGB images)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_image)

    if results.detections:
        for detection in results.detections:
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * width)
            y = int(bboxC.ymin * height)
            w = int(bboxC.width * width)
            h = int(bboxC.height * height)

            # # Adjust the y-coordinate and height to include more of the forehead
            # forehead_margin = int(0.2 * h)  # Adjust this percentage as needed
            # new_y = max(y - forehead_margin, 0)  # Ensure y does not go out of bounds
            # new_h = h + forehead_margin

            # Crop the adjusted face region
            face_image = image[y:y +h, x:x + w]

            # Save the cropped face
            cv2.imwrite(save_path, face_image)
            print(f"Cropped face saved to {save_path}")
            break  # Only process the first detected face
    else:
        print(f"No face detected in {image_path}!")


if __name__ == "__main__":
    emotions = ['ANG', 'FER', 'HPY', 'NUT', 'SAD', 'SUR']
    datasets = ['jaffe']

    crop_images('./new_dataset/Radboud_images/ANG/12_ANG.jpg', './cropped_face.png')
    # for dataset in datasets:
    #     for emotion in emotions:
    #         path_to_images = os.path.join(f"../dataset/data/{dataset}", emotion)
    #         if not os.path.exists(path_to_images):
    #             print(f"Error: Path {path_to_images} does not exist.")
    #             continue

    #         list_of_images = os.listdir(path_to_images)

    #         save_path = os.path.join(f"new_dataset/{dataset}", emotion)
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)

    #         for img in list_of_images:
    #             final_path = os.path.join(path_to_images, img)
    #             img_save_path = os.path.join(save_path, img)
    #             crop_images(final_path, img_save_path)

    #         print(f"Done for {emotion}. Number of images: {len(os.listdir(save_path))}")
        
    
# import os
# import cv2
# import mediapipe as mp
# import numpy as np

# def crop_images(image_path, save_path):
#     # Load Mediapipe Face Detection
#     mp_face_detection = mp.solutions.face_detection
#     face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

#     # Read the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Could not load image at {image_path}")
#         return

#     height, width, _ = image.shape
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_detection.process(rgb_image)

#     if results.detections:
#         for detection in results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             x = int(bboxC.xmin * width)
#             y = int(bboxC.ymin * height)
#             w = int(bboxC.width * width)
#             h = int(bboxC.height * height)

#             # Crop the face region
#             face_image = image[y:y + h, x:x + w]

#             # Create a circular mask
#             mask = np.zeros((h, w), dtype=np.uint8)
#             center = (w // 2, h // 2)
#             radius = min(w, h) // 2
#             cv2.circle(mask, center, radius, 255, -1)

#             # Apply the mask
#             result = cv2.bitwise_and(face_image, face_image, mask=mask)

#             # Make background transparent (optional)
#             face_with_alpha = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
#             face_with_alpha[:, :, 3] = mask

#             # Save the result
#             cv2.imwrite(save_path, face_with_alpha)
#             print(f"Cropped circular face saved to {save_path}")
#             break
#     else:
#         print(f"No face detected in {image_path}!")


# if __name__ == "__main__":
#     crop_images('./new_dataset/Radboud_images/ANG/12_ANG.jpg', './cropped_face.png')

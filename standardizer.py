import os
from PIL import Image

def resize_images_in_subdirectories(base_directory, size=(383,512)):
    # Walk through all subdirectories in the base directory
    for subdir, _, files in os.walk(base_directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            # Check if the file is an image
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff')):
                try:
                    with Image.open(file_path) as img:
                        # Resize the image using the LANCZOS resampling filter
                        img = img.resize(size, Image.Resampling.LANCZOS)
                        img.save(file_path)  # Overwrite the original image
                        print(f"Resized image: {file_path}")
                except Exception as e:
                    print(f"Failed to resize {file_path}: {e}")

# Example usage
base_directory = '/home/divyansh/repos/Facial-Emotions/landmark/IIM_GScale_Imgs'  # Update with your base directory path
resize_images_in_subdirectories(base_directory)
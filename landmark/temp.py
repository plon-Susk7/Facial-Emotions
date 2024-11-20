from helper import getLandmarkPoints, affine_trans, similarity_trans
from matplotlib import pyplot as plt
import numpy as np
import cv2 

image_path = 'landmark_point_sample.jpg'
image = cv2.imread(image_path)

h,w,_ = image.shape

# Get the landmark points and apply transformations
pointsA = getLandmarkPoints(image_path)
b1 = np.array(pointsA)
# _, points = affine_trans(np.array(pointsA))
# _, b1 = similarity_trans(points)

# scaled_points = b1 * np.array([1080, 1920])
plt.scatter(b1[:, 0], b1[:, 1])
plt.savefig("landmark_points.jpg")  # Save the scatter plot as an image
plt.show()  # Display the plot


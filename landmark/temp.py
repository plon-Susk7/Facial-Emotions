from helper import getLandmarkPoints, affine_trans, similarity_trans
from matplotlib import pyplot as plt
import numpy as np

image_path = 'data/radboud/ANG/Rafd090_01_Caucasian_female_angry_frontal.jpg'

# Get the landmark points and apply transformations
pointsA = getLandmarkPoints(image_path)
_, points = affine_trans(np.array(pointsA))
_, b1 = similarity_trans(points)

# Scatter plot of transformed points
plt.scatter(b1[:, 0], b1[:, 1])
plt.savefig("landmark_points_angry.jpg")  # Save the scatter plot as an image
plt.show()  # Display the plot

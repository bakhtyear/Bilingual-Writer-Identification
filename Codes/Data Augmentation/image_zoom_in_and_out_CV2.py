import os
import cv2
import numpy as np

# Set the source directory path
src_dir = "C:/Users/XENo/Desktop/All Data e"

# Define the range of zoom factors excluding 1
zoom_factors = np.concatenate((np.arange(0.90, 1.0, 0.02), np.arange(1.02, 1.11, 0.02)))

# Loop through all files in the source directory
for filename in os.listdir(src_dir):
    # Load the image
    img_path = os.path.join(src_dir, filename)
    img = cv2.imread(img_path)

    # Loop through the zoom factors and perform zoom in and zoom out
    for zoom_factor in zoom_factors:
        # Define the transformation matrix
        T = np.array([[zoom_factor, 0, 0], [0, zoom_factor, 0]])

        # Apply the transformation to the image
        transformed_img = cv2.warpAffine(img, T, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

        # Construct the output file name
        output_filename = f"{os.path.splitext(filename)[0]}_{zoom_factor:.2f}.jpg"

        # Save the transformed image
        cv2.imwrite(os.path.join(src_dir, output_filename), transformed_img)

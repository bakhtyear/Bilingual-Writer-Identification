import os
import cv2
import numpy as np

# Set the source directory path
src_dir = "C:/Users/XENo/Desktop/All Data"

# Loop through all files in the source directory
for filename in os.listdir(src_dir):
    # Load the image
    img_path = os.path.join(src_dir, filename)
    img = cv2.imread(img_path)

    # Define the range of degrees to rotate the image (exclude 0 degrees)
    degrees = np.concatenate((np.arange(-15, 0, 5), np.arange(5, 16, 5)))

    # Loop through the degrees and rotate the image
    for deg in degrees:
        # Define the rotation matrix
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), deg, 1)

        # Apply the rotation to the image
        rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255))

        # Construct the output file name
        output_filename = f"{os.path.splitext(filename)[0]}_{deg}.jpg"

        # Save the rotated image
        cv2.imwrite(os.path.join(src_dir, output_filename), rotated_img)

import os
import shutil

# Set the directory paths
src_dir = "C:/Users/XENo/Desktop/All Data"
dst_dir = "C:/Users/XENo/Desktop/Data Set 300"

# Loop through all files in the source directory
for filename in os.listdir(src_dir):
    # Extract the person ID from the filename (e.g. "1704001")
    person_id = filename.split("__")[0]

    # Construct the destination folder path
    dst_folder = os.path.join(dst_dir, person_id)

    # If the destination folder does not exist, create it
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Construct the source and destination file paths
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_folder, filename)

    # Move the file to the destination folder
    shutil.move(src_path, dst_path)

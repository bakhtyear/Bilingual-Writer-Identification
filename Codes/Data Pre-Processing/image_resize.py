from PIL import Image
import os

# Specify the folder containing your images
folder_path = "C:\\Users\\XENo\\Desktop\\All Data Bangla"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".png")):
        # Open the image
        img = Image.open(os.path.join(folder_path, filename))
        
        # Resize it to 32x32 pixels
        img = img.resize((32, 32))
        
        # Overwrite the original image with the resized version
        img.save(os.path.join(folder_path, filename))

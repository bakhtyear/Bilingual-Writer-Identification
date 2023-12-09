import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras

# Define the path to your saved model
model_path = r'C:\Users\XENo\Desktop\Thesis\your_model.h5'

# Define the path to your dataset
dataset_path = r'C:\Users\XENo\Desktop\Data Set'

# Create empty lists to store image data and labels
data = []
labels = []
class_mapping = {}  # Dictionary to map subfolders to class labels

# Traverse through the subdirectories and extract image data and labels
for class_idx, dir_name in enumerate(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, dir_name)

    # Loop through each image file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.jpg'):
            file_path = os.path.join(folder_path, file_name)

            # Open the image file using PIL and convert to grayscale
            with Image.open(file_path) as img:
                img_array = np.array(img)

            # Normalize pixel values to the range [0, 1]
            img_array = img_array / 255.0

            # Append image data and labels to the lists
            data.append(img_array)
            labels.append(class_idx)

            # Add mapping to the dictionary
            class_mapping[class_idx] = dir_name

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Load the saved model
model = keras.models.load_model(model_path)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Predict the labels for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate confusion matrix
confusion_mtx = confusion_matrix(y_test, y_pred_classes)


# Plot confusion matrix
#plt.figure(figsize=(10, 8))
#plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
#plt.title('Confusion Matrix')
#plt.colorbar()
#tick_marks = np.arange(len(class_mapping))
#plt.xticks(tick_marks, list(class_mapping.values()), rotation=90)
#plt.yticks(tick_marks, list(class_mapping.values()))
#plt.tight_layout()
#plt.ylabel('True Label')
#plt.xlabel('Predicted Label')
#plt.show()

# Save confusion matrix to a file
#np.savetxt("confusion_matrix.txt", confusion_mtx, fmt="%d")

# Plot confusion matrix as a heatmap
#custom_cmap = mcolors.ListedColormap(['black', 'red', 'white'])

plt.figure(figsize=(14, 12))
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
#cbar = plt.colorbar(cmap=custom_cmap)
plt.colorbar()
tick_marks = np.arange(len(class_mapping))
plt.xticks(tick_marks, list(class_mapping.values()), rotation=90)
plt.yticks(tick_marks, list(class_mapping.values()))
plt.tight_layout()

# Print values in each cell of the matrix
thresh = confusion_mtx.max() / 2.
for i in range(confusion_mtx.shape[0]):
    for j in range(confusion_mtx.shape[1]):
        plt.text(j, i, str(confusion_mtx[i, j]),
                 horizontalalignment="center",
                 color="white" if confusion_mtx[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Generate a classification report with precision, recall, F1 score, and support
class_report = classification_report(y_test, y_pred_classes, target_names=list(class_mapping.values()))
print("Classification Report:")
print(class_report)






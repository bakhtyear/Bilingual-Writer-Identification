import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet201  # Import DenseNet201
from tensorflow.keras.optimizers import Adam

# Define the path to your dataset
dataset_path = r'C:\Users\XENo\Desktop\Data Set'

# Create empty lists to store image data and labels
data = []
labels = []
class_mapping = {}

# Traverse through the subdirectories and extract image data and labels
for class_idx, dir_name in enumerate(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, dir_name)

    # Loop through each image file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.jpg'):
            file_path = os.path.join(folder_path, file_name)

            # Open the image file using PIL and resize to 32x32 pixels
            with Image.open(file_path) as img:
                img = img.resize((32, 32))
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

# Load the pre-trained DenseNet201 model with 32x32 input size
densenet_model = DenseNet201(weights='imagenet', input_shape=(32, 32, 3), include_top=False)

# Add your own classification layers on top of DenseNet201
densenet_model = keras.Sequential([
    densenet_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_mapping), activation='softmax')
])

# Compile the DenseNet201-based model
densenet_model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
densenet_model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))

# Evaluate the DenseNet201-based model
densenet_test_loss, densenet_test_accuracy = densenet_model.evaluate(X_test, y_test)
print("Test Accuracy (DenseNet201-based Model):", densenet_test_accuracy)

# Predict labels for DenseNet201-based model
y_pred_densenet = densenet_model.predict(X_test)
y_pred_classes_densenet = np.argmax(y_pred_densenet, axis=1)

# Compute precision, recall, and F1-score for DenseNet201-based model
precision_densenet = precision_score(y_test, y_pred_classes_densenet, average='weighted')
recall_densenet = recall_score(y_test, y_pred_classes_densenet, average='weighted')
f1_densenet = f1_score(y_test, y_pred_classes_densenet, average='weighted')

print("Precision (DenseNet201):", precision_densenet)
print("Recall (DenseNet201):", recall_densenet)
print("F1 Score (DenseNet201):", f1_densenet)
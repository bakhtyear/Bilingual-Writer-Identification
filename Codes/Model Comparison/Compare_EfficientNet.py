import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0  # Import EfficientNetB0
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

# Load the pre-trained EfficientNetB0 model with 32x32 input size
efficientnet_model = EfficientNetB0(weights='imagenet', input_shape=(32, 32, 3), include_top=False)

# Add your own classification layers on top of EfficientNetB0
efficientnet_model = keras.Sequential([
    efficientnet_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_mapping), activation='softmax')
])

# Compile the EfficientNetB0-based model
efficientnet_model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the EfficientNetB0-based model
efficientnet_model.fit(X_train, y_train, epochs=8, batch_size=128, validation_data=(X_test, y_test))

# Evaluate the EfficientNetB0-based model
efficientnet_test_loss, efficientnet_test_accuracy = efficientnet_model.evaluate(X_test, y_test)
print("Test Accuracy (EfficientNetB0-based Model):", efficientnet_test_accuracy)

# Predict labels for EfficientNetB0-based model
y_pred_efficientnet = efficientnet_model.predict(X_test)
y_pred_classes_efficientnet = np.argmax(y_pred_efficientnet, axis=1)

# Compute precision, recall, and F1-score for EfficientNetB0-based model
precision_efficientnet = precision_score(y_test, y_pred_classes_efficientnet, average='weighted')
recall_efficientnet = recall_score(y_test, y_pred_classes_efficientnet, average='weighted')
f1_efficientnet = f1_score(y_test, y_pred_classes_efficientnet, average='weighted')

print("Precision (EfficientNetB0):", precision_efficientnet)
print("Recall (EfficientNetB0):", recall_efficientnet)
print("F1 Score (EfficientNetB0):", f1_efficientnet)

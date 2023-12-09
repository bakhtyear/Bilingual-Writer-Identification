import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3  # Import InceptionV3
from tensorflow.keras.optimizers import Adam

# Define the path to your dataset
dataset_path = r'C:\Users\XENo\Desktop\Data Set 300'

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
                img = img.resize((75, 75))
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

# Load the pre-trained InceptionV3 model with 32x32 input size
inception_model = InceptionV3(weights='imagenet', input_shape=(75, 75, 3), include_top=False)

# Add your own classification layers on top of InceptionV3
inception_model = keras.Sequential([
    inception_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_mapping), activation='softmax')
])

# Compile the InceptionV3-based model
inception_model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the InceptionV3-based model
inception_model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the InceptionV3-based model
inception_test_loss, inception_test_accuracy = inception_model.evaluate(X_test, y_test)
print("Test Accuracy (InceptionV3-based Model):", inception_test_accuracy)

# Predict labels for InceptionV3-based model
y_pred_inception = inception_model.predict(X_test)
y_pred_classes_inception = np.argmax(y_pred_inception, axis=1)

# Compute precision, recall, and F1-score for InceptionV3-based model
precision_inception = precision_score(y_test, y_pred_classes_inception, average='weighted')
recall_inception = recall_score(y_test, y_pred_classes_inception, average='weighted')
f1_inception = f1_score(y_test, y_pred_classes_inception, average='weighted')

print("Precision (InceptionV3):", precision_inception)
print("Recall (InceptionV3):", recall_inception)
print("F1 Score (InceptionV3):", f1_inception)

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

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

            # Assuming images are resized to 32x32 pixels
            img = keras.preprocessing.image.load_img(file_path, target_size=(32, 32))
            img_array = keras.preprocessing.image.img_to_array(img)

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

# Convert labels to one-hot encoding
labels_one_hot = keras.utils.to_categorical(labels, num_classes=len(class_mapping))

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)

# Define the CNN model for a 32x32 input size
cnn_model = keras.Sequential([
    # Convolutional layer 1
    layers.Conv2D(96, (5, 5), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2), strides=(2, 2)),
    
    # Convolutional layer 2
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), strides=(2, 2)),
    
    # Convolutional layers 3-5
    layers.Conv2D(384, (3, 3), activation='relu'),
    layers.Conv2D(384, (3, 3), activation='relu'),
    layers.Conv2D(256, (3, 3), activation='relu'),
    
    # Flatten layer
    layers.Flatten(),
    
    # Fully connected layers 6-8
#    layers.Dense(4096, activation='relu'),
#    layers.Dropout(0.5),
#    layers.Dense(4096, activation='relu'),
#    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    # Output layer
    layers.Dense(len(class_mapping), activation='softmax')
])


# Compile the model
cnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Predict labels for the test set
y_pred = cnn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate precision, recall, and F1-score
precision = precision_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')
recall = recall_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')
f1 = f1_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

import glob

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

if tf.test.gpu_device_name():
    print("Default GPU Device:", tf.test.gpu_device_name())
else:
    print("GPU not found. Using CPU.")

file_path = 'DataPreparation/merged_data.csv'

all_features = []
all_labels = []

scaler = StandardScaler()

chunk_size = 21 * 10
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    num_samples = len(chunk) // 21
    reshaped_chunk = chunk.values.reshape(num_samples, 21, -1)
    X_chunk = reshaped_chunk[:, :, :-1].reshape(num_samples, -1)
    y_chunk = reshaped_chunk[:, 0, -1]
    all_features.append(X_chunk)
    all_labels.append(y_chunk)

all_features = np.vstack(all_features)
all_features_scaled = scaler.fit_transform(all_features)

label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(np.concatenate(all_labels))
num_emotions = len(np.unique(all_labels_encoded))
all_labels_categorical = tf.keras.utils.to_categorical(all_labels_encoded, num_classes=num_emotions)

X_train, X_test, y_train, y_test = train_test_split(all_features_scaled, all_labels_categorical, test_size=0.2,
                                                    random_state=42)

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(num_emotions, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

model.save('/Models/HandLandmarksModelDNN')

test_loss, test_accuracy_DNN = model.evaluate(X_test, y_test)
print("Test DNN Accuracy :", test_accuracy_DNN)
#
img_size = (128, 128)  # Example size, can be adjusted
num_emotions = 6  # Number of classes

# Load images and labels
images = []
labels = []

for img_path in glob.glob('/DataPreparation/heatmaps/*.png'):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)

    img = img / 255.0

    label = img_path.split('_')[-2]

    images.append(img)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)
print(images.shape)
print(labels.shape)
print(np.unique(labels))

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
one_hot_encoded = to_categorical(integer_encoded, num_classes=num_emotions)

X_train, X_test, y_train, y_test = train_test_split(images, one_hot_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_emotions, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

test_loss, test_accuracy_CNN = model.evaluate(X_test, y_test)
print("Test  CNN accuracy:", test_accuracy_CNN)
# print("Test DNN Accuracy :", test_accuracy_DNN)

model.save('/Models/HandLandmarksModelCNN')

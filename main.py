import glob
import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, SimpleRNN, Dropout, LSTM
from tensorflow.keras.models import Sequential, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if tf.test.gpu_device_name():
    print("Default GPU Device:", tf.test.gpu_device_name())
else:
    print("GPU not found. Using CPU.")

file_path = 'D:/ML/Diploma_work/DataPreparation/merged_data.csv'
df = pd.read_csv(file_path)
all_features = []
all_labels = []

scaler = StandardScaler()

chunk_size = 21 * 100
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

# dnn_model = Sequential([
#     Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     Dropout(0.5),
#     Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     Dropout(0.5),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(num_emotions, activation='softmax')
# ])
#
# dnn_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#
# dnn_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping])
dnn_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_emotions, activation='softmax')
])

# Inicjalna szybkość uczenia
initial_learning_rate = 0.01

# Funkcja adaptacyjnej szybkości uczenia
def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 10
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

# Kompilacja modelu z optymalizatorem Adam
dnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Dodanie LearningRateScheduler do callbacks
callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]

# Trenowanie modelu
history = dnn_model.fit(X_train, y_train, epochs=50, batch_size=64,
                    validation_data=(X_test, y_test), callbacks=callbacks)

# Ewaluacja modelu
test_loss, test_accuracy = dnn_model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)
dnn_model.save('D:/ML/Diploma_work/Models/HandLandmarksModelDNN_ver2')

test_loss, test_accuracy_DNN = dnn_model.evaluate(X_test, y_test)
print("Test DNN Accuracy :", test_accuracy_DNN)

y_pred = dnn_model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test_labels, y_pred_labels)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=label_encoder.classes_, title='Confusion matrix')
plt.savefig('dnn_confusion_matrix.png')

fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('dnn_roc_curve.png')

precision, recall, _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())

plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.savefig('dnn_precision_recall_curve.png')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

fcnn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(num_emotions, activation='softmax')
])

fcnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fcnn_model.fit(X_train, y_train, batch_size=32, epochs=15, validation_split=0.2, callbacks=[early_stopping])

test_loss, test_accuracy_FCNN = fcnn_model.evaluate(X_test, y_test)
print("FCNN Test accuracy:", test_accuracy_FCNN)

fcnn_model.save('D:/ML/Diploma_work/Models/HandLandmarksModelFCNN')

y_pred = fcnn_model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test_labels, y_pred_labels)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=label_encoder.classes_, title='Confusion matrix')
plt.savefig('fcnn_confusion_matrix.png')

fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('fcnn_roc_curve.png')

precision, recall, _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())

plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.savefig('fcnn_precision_recall_curve.png')

X_train_rnn = X_train.reshape(-1, 21, X_train.shape[1] // 21)
X_test_rnn = X_test.reshape(-1, 21, X_test.shape[1] // 21)

input_shape = (X_train_rnn.shape[1], X_train_rnn.shape[2])

lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=input_shape),
    Dropout(0.5),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(num_emotions, activation='softmax')
])

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lstm_model.fit(X_train_rnn, y_train, batch_size=32, epochs=15, validation_split=0.2)

test_loss_LSTM, test_accuracy_LSTM = lstm_model.evaluate(X_test_rnn, y_test)
print("Test LSTM accuracy:", test_accuracy_LSTM)

lstm_model.save('D:/ML/Diploma_work/Models/HandLandmarksModelLSTM')

y_pred = lstm_model.predict(X_test_rnn)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test_labels, y_pred_labels)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=label_encoder.classes_, title='Confusion matrix - LSTM')
plt.savefig('lstm_confusion_matrix.png')

# Krzywa ROC
fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('rnn_roc_curve.png')

precision, recall, _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())

plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.savefig('rnn_precision_recall_curve.png')


def load_images_in_chunks(file_paths, chunk_size, img_size=(64, 64)):
    chunk_images = []
    chunk_labels = []

    for i, img_path in enumerate(file_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        img = (img / 255.0).astype('float32')

        label = img_path.split('_')[-2]
        chunk_images.append(img)
        chunk_labels.append(label)

        if (i + 1) % chunk_size == 0 or (i + 1) == len(file_paths):
            yield np.array(chunk_images), np.array(chunk_labels)
            chunk_images = []
            chunk_labels = []


img_paths = glob.glob('E:/Datasety/heatmaps/*.png')
chunk_size = 5
num_emotions = 6

all_images = []
all_labels = []

for images, labels in load_images_in_chunks(img_paths, chunk_size):
    all_images.extend(images)
    all_labels.extend(labels)

all_images = np.array(all_images)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(all_labels)
one_hot_encoded = to_categorical(integer_encoded, num_classes=num_emotions)

X_train, X_test, y_train, y_test = train_test_split(all_images, one_hot_encoded, test_size=0.2, random_state=42)

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_emotions, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn_model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test), callbacks=[early_stopping])

test_loss, test_accuracy_CNN = cnn_model.evaluate(X_test, y_test)
print("Test CNN accuracy:", test_accuracy_CNN)

cnn_model.save('D:/ML/Diploma_work/Models/HandLandmarksModelCNN')

y_pred = cnn_model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test_labels, y_pred_labels)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=label_encoder.classes_, title='Confusion matrix')
plt.savefig('cnn_confusion_matrix.png')

fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('cnn_roc_curve.png')

precision, recall, _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())

plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.savefig('cnn_precision_recall_curve.png')

print("Test DNN Accuracy :", test_accuracy_DNN)
print("Test CNN Accuracy :", test_accuracy_CNN)
print("Test FCNN Accuracy :", test_accuracy_FCNN)
print("Test RNN Accuracy :", test_accuracy_LSTM)

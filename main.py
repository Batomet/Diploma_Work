import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


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

dnn_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_emotions, activation='softmax')
])

dnn_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

dnn_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

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

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Przygotowanie etykiet dla modelu Random Forest
y_train_rf = np.argmax(y_train, axis=1)
y_test_rf = np.argmax(y_test, axis=1)

# Trenowanie modelu na danych treningowych
rf_model.fit(X_train, y_train_rf)

# Predykcja na danych testowych
y_pred_rf = rf_model.predict(X_test)

# Ocena modelu
test_accuracy_RF = accuracy_score(y_test_rf, y_pred_rf)
print("Random Forest Test accuracy:", test_accuracy_RF)
dump(rf_model, 'D:/ML/Diploma_work/Models/HandLandmarksModelRF')

# Macierz pomyłek
cm_rf = confusion_matrix(y_test_rf, y_pred_rf)
plt.figure()
plot_confusion_matrix(cm_rf, classes=label_encoder.classes_, title='Random Forest Confusion Matrix')
plt.savefig('rf_confusion_matrix.png')

# Krzywa ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test.ravel(), rf_model.predict_proba(X_test).ravel())
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='Random Forest ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('rf_roc_curve.png')

precision, recall, _ = precision_recall_curve(y_test_rf.ravel(), y_pred_rf.ravel())

plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.savefig('RF_precision_recall_curve.png')

print("Test DNN Accuracy :", test_accuracy_DNN)
print("Test RF Accuracy :", test_accuracy_RF)
print("Test FCNN Accuracy :", test_accuracy_FCNN)
print("Test RNN Accuracy :", test_accuracy_LSTM)

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')



#dataset loading with file filtering
paths = []
labels = []
valid_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.aiff')  # Supported audio formats

for dirname, _, filenames in os.walk('/content/drive/MyDrive/TESS Toronto emotional speech set data copy'):
    for filename in filenames:
        # Skip non-audio files (like .DS_Store)
        if filename.lower().endswith(valid_extensions):
            paths.append(os.path.join(dirname, filename))
            # Improved label extraction for TESS dataset
            label = filename.split('_')[-1].split('.')[0].lower()
            labels.append(label)

print(f"Dataset loaded with {len(paths)} valid audio files")

df = pd.DataFrame({'speech': paths, 'label': labels})
print("\nLabel distribution:")
print(df['label'].value_counts())

# Visualizations
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='label')
plt.title('Emotion Label Distribution')
plt.xticks(rotation=45)
plt.show()



# Audio visualization functions
def plot_audio_analysis(emotion, df):
    try:
        path = np.array(df['speech'][df['label'] == emotion])[0]
        data, sampling_rate = librosa.load(path)

        plt.figure(figsize=(15, 5))
        plt.suptitle(f'Analysis for: {emotion}', y=1.05)

        # Waveplot
        plt.subplot(1, 2, 1)
        librosa.display.waveshow(data, sr=sampling_rate)
        plt.title('Waveform')

        # Spectrogram
        plt.subplot(1, 2, 2)
        x = librosa.stft(data)
        xdb = librosa.amplitude_to_db(abs(x))
        librosa.display.specshow(xdb, sr=sampling_rate, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')

        plt.tight_layout()
        plt.show()

        return Audio(path)
    except Exception as e:
        print(f"Error processing {emotion}: {str(e)}")
        return None



# Analyze sample emotions
emotions_to_analyze = ['ps', 'happy', 'fear', 'disgust', 'neutral', 'sad', 'angry']
for emotion in emotions_to_analyze:
    display(plot_audio_analysis(emotion, df))



# Feature extraction with error handling
def extract_MFCC(filename):
    try:
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None



print("\nExtracting MFCC features...")
X_mfcc = df['speech'].apply(extract_MFCC)



# Remove any None values from failed processing
valid_indices = [i for i, x in enumerate(X_mfcc) if x is not None]
df = df.iloc[valid_indices]
X_mfcc = [x for x in X_mfcc if x is not None]

X = np.array(X_mfcc)
X = np.expand_dims(X, -1)



# Label encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(df['label'])
y = OneHotEncoder(sparse_output=False).fit_transform(y_encoded.reshape(-1, 1))



# LSTM Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

model = Sequential([
    LSTM(128, return_sequences=False, input_shape=(40, 1)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Add early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nTraining model...")
history = model.fit(X, y,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=512,
                    shuffle=True,
                    callbacks=[early_stop])



# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()



# Evaluation
from sklearn.metrics import confusion_matrix, classification_report
from numpy import argmax

y_pred = model.predict(X)
y_pred_classes = argmax(y_pred, axis=1)
y_true_classes = argmax(y, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()




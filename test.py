import os
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize, LabelEncoder, OneHotEncoder
import seaborn as sns
from numpy import argmax
from itertools import cycle
import joblib
from keras.models import load_model

def load_dataset(audio_dir):
    print(f"\nChecking directory: {audio_dir}")

    # Verify directory exists
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Directory not found: {audio_dir}")

    paths = []
    labels = []
    supported_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.aiff', '.m4a')
    skipped_files = 0

    for dirpath, _, filenames in os.walk(audio_dir):
        for filename in filenames:
            # Skip hidden files and system files
            if filename.startswith('.'):
                skipped_files += 1
                continue

            # Check file extension
            if not filename.lower().endswith(supported_extensions):
                skipped_files += 1
                continue

            full_path = os.path.join(dirpath, filename)

            # Verify file is accessible
            if not os.path.exists(full_path):
                print(f"Warning: File not accessible - {full_path}")
                skipped_files += 1
                continue

            try:
                # Extract label from filename
                base_name = os.path.splitext(filename)[0]
                label = base_name.split('_')[-1].lower()
                paths.append(full_path)
                labels.append(label)
            except Exception as e:
                print(f"Warning: Couldn't process filename {filename} - {str(e)}")
                skipped_files += 1
                continue

    if not paths:
        print("\nDebug Info:")
        print(f"Total files scanned: {len(os.listdir(audio_dir))}")
        print(f"Files skipped: {skipped_files}")
        print(f"Supported extensions: {supported_extensions}")
        raise ValueError("No valid audio files found. Check directory path and file formats.")

    print(f"\nSuccessfully loaded {len(paths)} audio files")
    print(f"Skipped {skipped_files} non-audio files")
    return pd.DataFrame({'speech': paths, 'label': labels})

def extract_features(df):
    """Extract MFCC features with comprehensive error handling"""
    features = []
    failed_files = []

    for idx, filepath in enumerate(df['speech']):
        try:
            # Load audio with librosa (3 seconds max, starting at 0.5s)
            y, sr = librosa.load(filepath, duration=3, offset=0.5, sr=22050)

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc = np.mean(mfcc.T, axis=0)

            # Validate feature shape
            if mfcc.shape[0] != 40:
                raise ValueError(f"Invalid MFCC shape: {mfcc.shape}")

            features.append(mfcc)
        except Exception as e:
            print(f"Error processing {os.path.basename(filepath)}: {str(e)}")
            failed_files.append(filepath)
            continue

    if not features:
        raise ValueError("Feature extraction failed for all files!")

    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files")
        df = df[~df['speech'].isin(failed_files)].reset_index(drop=True)

    return np.array(features), df

def evaluate_model(model, X, y, le, df):
    """Comprehensive model evaluation with visualizations"""
    # Predictions
    y_pred = model.predict(X)
    y_pred_classes = argmax(y_pred, axis=1)
    y_true_classes = argmax(y, axis=1)
    class_names = le.classes_

    # Metrics
    print("\nEvaluation Metrics:")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_true_classes, y_pred_classes):.4f}")
    print(f"Precision (weighted): {precision_score(y_true_classes, y_pred_classes, average='weighted'):.4f}")
    print(f"Recall (weighted): {recall_score(y_true_classes, y_pred_classes, average='weighted'):.4f}")
    print(f"F1-Score (weighted): {f1_score(y_true_classes, y_pred_classes, average='weighted'):.4f}")

    # Classification Report
    print("\nClassification Report:")
    print("="*50)
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # ROC Curve (for multi-class)
    if len(class_names) <= 10:
        y_test_bin = label_binarize(y_true_classes, classes=np.unique(y_true_classes))
        n_classes = y_test_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        plt.figure(figsize=(10, 8))

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
        for i, color in zip(range(n_classes), colors):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    # Error Analysis
    errors = y_pred_classes != y_true_classes
    error_indices = np.where(errors)[0]

    if len(error_indices) > 0:
        print(f"\nMisclassified Samples ({len(error_indices)}/{len(y_true_classes)}):")
        for i in error_indices[:5]:  # Show first 5 errors
            print(f"\nFile: {os.path.basename(df.iloc[i]['speech'])}")
            print(f"True: {class_names[y_true_classes[i]]}")
            print(f"Predicted: {class_names[y_pred_classes[i]]}")
    else:
        print("\nPerfect classification on all samples!")

def main():
    """Main execution flow"""
    try:
        # 1. Load Dataset
        audio_dir = '/content/drive/MyDrive/TestModel'
        print("\n" + "="*50)
        print("Loading Audio Dataset")
        print("="*50)
        df = load_dataset(audio_dir)

        # Show label distribution
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x='label')
        plt.title('Label Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # 2. Feature Extraction
        print("\n" + "="*50)
        print("Extracting Audio Features")
        print("="*50)
        X, df = extract_features(df)
        X = np.expand_dims(X, -1)  # Add channel dimension for LSTM

        # 3. Label Encoding
        le = LabelEncoder()
        y_encoded = le.fit_transform(df['label'])
        enc = OneHotEncoder(sparse_output=False)
        y = enc.fit_transform(y_encoded.reshape(-1, 1))

        # 4. Load Model
        print("\n" + "="*50)
        print("Loading Pre-trained Model")
        print("="*50)
        model_path = '/content/emotion_sensor.h5'
        model = load_model(model_path)

        # 5. Model Evaluation
        print("\n" + "="*50)
        print("Evaluating Model Performance")
        print("="*50)
        evaluate_model(model, X, y, le, df)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting Tips:")
        print("- Verify the audio directory path is correct")
        print("- Check files have supported extensions (.wav, .mp3, etc.)")
        print("- Ensure model file exists at specified path")
        print("- Confirm files follow naming convention: name_emotion.ext")

if __name__ == "__main__":
    main()
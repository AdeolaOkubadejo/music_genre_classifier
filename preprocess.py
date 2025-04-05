import librosa
import numpy as np
import pandas as pd
import os

def extract_features(file_path):
    y, sr = librosa.load(file_path)  # Works with .au or .wav
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    tempo = librosa.beat.tempo(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr), sr=sr)  # Updated tempo extraction
    return np.hstack([mfcc, tempo])

data = []
labels = []
base_path = "genres_original"  # Matches your Kaggle folder

for genre in os.listdir(base_path):
    genre_path = os.path.join(base_path, genre)
    if os.path.isdir(genre_path):  # Ensure it’s a folder
        print(f"Processing genre: {genre}")
        for file in os.listdir(genre_path):
            if file.endswith(".au"):  # Check for .au files
                file_path = os.path.join(genre_path, file)
                print(f"Found file: {file_path}")
                features = extract_features(file_path)
                data.append(features)
                labels.append(genre)

df = pd.DataFrame(data)
df["label"] = labels
df.to_csv("features.csv", index=False)
print("Features saved to features.csv!")
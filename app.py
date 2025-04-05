from flask import Flask, request, render_template
import joblib
import librosa
import numpy as np
import os
import requests

app = Flask(__name__)

# Download the model file if it doesn’t exist
model_url = "https://drive.google.com/uc?export=download&id=1vOMTJz8JtXwFMvEek1e4gFRuJYl_nZAv"  # Replace with your actual link
model_path = "genre_classifier.pkl"
if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
model = joblib.load(model_path)

# Create uploads folder if it doesn’t exist
uploads_dir = "uploads"
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    tempo = librosa.beat.tempo(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr), sr=sr)
    return np.hstack([mfcc, tempo])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        file_path = os.path.join(uploads_dir, file.filename)  # Use uploads_dir here
        file.save(file_path)
        features = extract_features(file_path)
        prediction = model.predict([features])[0]
        probs = model.predict_proba([features])[0]
        genres = model.classes_
        return render_template("result.html", genre=prediction, probs=probs.tolist(), genres=genres.tolist())
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template
import joblib
import librosa
import numpy as np
import os

app = Flask(__name__)
model = joblib.load("genre_classifier.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    tempo = librosa.beat.tempo(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr), sr=sr)
    return np.hstack([mfcc, tempo])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        features = extract_features(file_path)
        prediction = model.predict([features])[0]
        probs = model.predict_proba([features])[0]
        genres = model.classes_
        return render_template("result.html", genre=prediction, probs=probs.tolist(), genres=genres.tolist())
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
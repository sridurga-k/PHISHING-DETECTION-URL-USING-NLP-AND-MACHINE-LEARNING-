from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load model and vectorizer (must be in the same folder)
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("url")
    if not url:
        return jsonify({"result": "Invalid URL"})

    url_vector = vectorizer.transform([url])
    prediction = model.predict(url_vector)[0]
    result = "Phishing" if prediction == 1 else "Legitimate"
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)

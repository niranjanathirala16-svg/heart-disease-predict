from flask import Flask, render_template, request

import joblib


app = Flask(__name__)

# Load trained Random Forest model
model = joblib.load("heart_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["id"]),
        float(request.form["age"]),
        float(request.form["gender"]),
        float(request.form["height"]),
        float(request.form["weight"]),
        float(request.form["ap_hi"]),
        float(request.form["ap_lo"]),
        float(request.form["cholesterol"]),
        float(request.form["glucose"]),
        float(request.form["smoke"]),
        float(request.form["alco"]),
        float(request.form["active"])
    ]

    prediction = model.predict([features])

    if prediction[0] == 1:
        result = "Heart disease detected"
    else:
        result = "No heart disease detected"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
 app.run(host="0.0.0.0", port=10000)


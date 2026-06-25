from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load Trained Model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        age = int(request.form["age"])
        income = int(request.form["income"])
        experience = int(request.form["experience"])
        score = int(request.form["score"])

        data = np.array([[age, income, experience, score]])

        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "Customer Will Purchase"
        else:
            result = "Customer Will Not Purchase"

        return render_template("result.html", result=result)

    except Exception as e:
        return render_template("result.html", result=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# ==== Load Vectorizer ====
with open("app/models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ==== Load All Models ====
model_paths = {
    "nb": "app/models/best_nb_model.pkl",
    "svm": "app/models/best_svm_model.pkl",
    "rf": "app/models/best_rf_model.pkl",
    "xgb": "app/models/best_xgb_model.pkl",
    "lr": "app/models/best_lr_model.pkl"
}

models = {}
for key, path in model_paths.items():
    with open(path, "rb") as f:
        models[key] = pickle.load(f)

print("✅ Models and vectorizer loaded successfully!")

# ==== Helper Function ====
def get_prediction_results(url_text):
    """
    Takes the input URL, vectorizes it, and predicts using all 5 models.
    Returns a dictionary of results.
    """
    X = vectorizer.transform([url_text])
    results = {}

    # Predict using each model
    for key, model in models.items():
        pred = model.predict(X)[0]
        results[key] = "Phishing" if pred == 1 else "Legitimate"

    # Majority vote for overall result
    votes = list(results.values())
    phishing_votes = votes.count("Phishing")
    overall = "Phishing Website 🚨" if phishing_votes > len(votes) / 2 else "Legitimate Website ✅"

    results["overall"] = overall
    return results


# ==== Routes ====
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    url_text = request.form["text"]
    predictions = get_prediction_results(url_text)

    # Render the HTML with all prediction results
    return render_template(
        "index.html",
        data=url_text,
        prediction_result=predictions["overall"],
        nb_result=predictions["nb"],
        svm_result=predictions["svm"],
        rf_result=predictions["rf"],
        xgb_result=predictions["xgb"],
        lr_result=predictions["lr"],
    )


if __name__ == "__main__":
    app.run(debug=True)

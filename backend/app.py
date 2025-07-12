from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

# import the class so unpickling works
from components import FeatureEngineer

app = Flask(__name__)
CORS(app)

pipeline = joblib.load("model/titanic_pipeline.pkl")
print("✅ Loaded Titanic pipeline.")

@app.route("/")
def home():
    return "Titanic Survival API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    prob = pipeline.predict_proba(input_df)[0][1]
    pred = pipeline.predict(input_df)[0]
    return jsonify({
        "survival_probability": round(float(prob), 4),
        "survived": int(pred)
    })

if __name__ == "__main__":
    app.run(debug=True)
import pandas as pd
from flask import Flask, request, jsonify
import pickle
from dotenv import load_dotenv
import os
from groq import Groq

# Initialize Flask app
flask_app = Flask(__name__)

groq_client = None
models = []

def load_models():

    global models

    filenames = [("XGBoost", "./models/xgb_model.pkl"), ("Random Forest", "./models/rf_model.pkl"), ("K-Nearest Neighbors", "./models/knn_model.pkl")]
    
    for model_name, filename in filenames:
        with open(filename, "rb") as file:
            model = pickle.load(file)
            models.append((model_name, model))

@flask_app.route("/get-data", methods=["GET"])
def get_data():
    
    try:
        
        df = pd.read_csv("./db/churn.csv")
        
        churn_data = df.to_dict(orient="records")
        
        return jsonify(churn_data), 200

    except Exception as e:
        
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@flask_app.route("/predict", methods=["POST"])
def predict():
    
    if request.method == "POST":
        
        input_dict = request.json
        
        input_df = pd.DataFrame([input_dict])
        
        probabilities = {}
        for model_name, model in models:
            prob = model.predict_proba(input_df)[0][1]
            probabilities[model_name] = float(prob)

        return jsonify(probabilities), 200
    
    else:
        
        return jsonify({"error": "Method not allowed"}), 405

@flask_app.route("/explain-prediction", methods=["POST"])
def explain_prediction():

    if request.method == "POST":

        data = request.json

        system_prompt = data["system_prompt"]
        prompt = data["prompt"]

        raw_response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        return jsonify({ "response": raw_response.choices[0].message.content }), 200

    else:

        return jsonify({"error": "Method not allowed"}), 405

@flask_app.route("/generate-email", methods=["POST"])
def generate_email():

    if request.method == "POST":

        data = request.json

        system_prompt = data["system_prompt"]
        prompt = data["prompt"]

        raw_response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        return jsonify({ "response": raw_response.choices[0].message.content }), 200

    else:

        return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':

    # Load environment variables
    load_dotenv()

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Load model
    load_models() 

    # Run app
    flask_app.run(host='0.0.0.0', port=8000)

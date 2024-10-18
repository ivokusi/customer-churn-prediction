import pandas as pd
from flask import Flask, request, jsonify
import pickle
from dotenv import load_dotenv
from groq import Groq
import os

flask_app = Flask(__name__)

flask_app.config["groq_client"] = None
flask_app.config["models"] = []
flask_app.config["model_load_error"] = None

def load_models():

    try:

        # Get the absolute path to the directory containing this script
        base_path = os.path.dirname(os.path.abspath(__file__))

        filenames = [
            ("XGBoost", os.path.join(base_path, "models", "xgb_model.pkl")),
            ("Random Forest", os.path.join(base_path, "models", "rf_model.pkl")),
            ("K-Nearest Neighbors", os.path.join(base_path, "models", "knn_model.pkl"))
        ]

        for model_name, filename in filenames:
            with open(filename, "rb") as file:
                model = pickle.load(file)
                flask_app.config["models"].append((model_name, model))
        
    except Exception as e:
        
        flask_app.config["model_load_error"] = str(e)

@flask_app.route("/get-data", methods=["GET"])
def get_data():
    
    try:
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(os.path.join(base_path, "db", "churn.csv"))
        
        churn_data = df.to_dict(orient="records")
        
        return jsonify(churn_data), 200

    except Exception as e:
        
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@flask_app.route("/predict", methods=["POST"])
def predict():

    if flask_app.config["model_load_error"] is not None:

        return jsonify({"error": flask_app.config["model_load_error"]}), 500
    
    if request.method == "POST":
        
        input_dict = request.json
        
        input_df = pd.DataFrame([input_dict])
        
        probabilities = {}
        for model_name, model in flask_app.config["models"]:
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

        raw_response = flask_app.config["groq_client"].chat.completions.create(
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

        try:

            data = request.json

            system_prompt = data["system_prompt"]
            prompt = data["prompt"]

            raw_response = flask_app.config["groq_client"].chat.completions.create(
                model="llama-3.1-70b-versatile", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )

            return jsonify({ "response": raw_response.choices[0].message.content }), 200
    
        except Exception as e:

            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    else:

        return jsonify({"error": "Method not allowed"}), 405

@flask_app.route("/", methods=["GET"])
def test():
    
    return jsonify({}), 200

if __name__ == '__main__':

    # Load environment variables
    load_dotenv()

    flask_app.config["groq_client"] = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Load model
    load_models()

    # Run app
    flask_app.run(host='0.0.0.0', port=8000)

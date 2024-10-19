import pandas as pd
from flask import Flask, request, jsonify
import pickle
from groq import Groq
import os

flask_app = Flask(__name__)

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
    
    if request.method == "POST":

        base_path = os.path.dirname(os.path.abspath(__file__))

        filenames = [
            ("XGBoost", os.path.join(base_path, "models", "xgb_model.pkl")),
            ("Random Forest", os.path.join(base_path, "models", "rf_model.pkl")),
            ("K-Nearest Neighbors", os.path.join(base_path, "models", "knn_model.pkl"))
        ]

        models = []

        for model_name, filename in filenames:
            with open(filename, "rb") as file:
                model = pickle.load(file)
                models.append((model_name, model))
        
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

        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        data = request.json

        system_prompt = data["system_prompt"]
        prompt = data["prompt"]

        models = ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it"]

        for model in models:

            try:

                raw_response = groq_client.chat.completions.create(
                    model=model, 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )

                break

            except Exception as e:

                continue

        return jsonify({ "response": raw_response.choices[0].message.content }), 200

    else:

        return jsonify({"error": "Method not allowed"}), 405

@flask_app.route("/generate-email", methods=["POST"])
def generate_email():

    if request.method == "POST":

        try:

            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            data = request.json

            system_prompt = data["system_prompt"]
            prompt = data["prompt"]

            models = ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it"]

            for model in models:

                try:

                    raw_response = groq_client.chat.completions.create(
                        model=model, 
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                    )

                    break

                except Exception as e:

                    continue

            return jsonify({ "response": raw_response.choices[0].message.content }), 200
    
        except Exception as e:

            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    else:

        return jsonify({"error": "Method not allowed"}), 405

@flask_app.route("/", methods=["GET"])
def test():
    
    return jsonify({}), 200

if __name__ == '__main__':

    flask_app.run(host='0.0.0.0', port=8000)

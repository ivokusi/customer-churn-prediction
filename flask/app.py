import pandas as pd
from flask import Flask, request, jsonify
import pickle
from dotenv import load_dotenv
from groq import Groq
import os

flask_app = Flask(__name__)

groq_client = None
models = []
load_error = None

def load_models():

    global models
    global load_error

    try:

        filenames = [("XGBoost", "./models/xgb_model.pkl"), ("Random Forest", "./models/rf_model.pkl"), ("K-Nearest Neighbors", "./models/knn_model.pkl")]
        
        for model_name, filename in filenames:
            with open(filename, "rb") as file:
                model = pickle.load(file)
                models.append((model_name, model))
        
    except Exception as e:
        
        load_error = str(e)

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

    if load_error:

        return jsonify({"error": load_error}), 500
    
    if request.method == "POST":
        
        input_dict = request.json
        
        input_df = pd.DataFrame([input_dict])
        
        probabilities = {}
        for model_name, model in models:
            prob = model.predict_proba(input_df)[0][1]
            probabilities[model_name] = float(prob)

        return jsonify(probabilities, len(models)), 200
    
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

        try:

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
    
        except Exception as e:

            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    else:

        return jsonify({"error": "Method not allowed"}), 405

@flask_app.route("/", methods=["GET"])
def test():
    current_directory = os.getcwd()
    
    # Get files in current directory
    current_files = os.listdir('.')
    
    # Get files in 'db' directory
    db_files = os.listdir('./db')
    
    # Get files in 'models' directory
    models_files = os.listdir('./models')
    
    return jsonify({
        "current_directory": current_directory,
        "current_files": current_files,
        "db_files": db_files,
        "models_files": models_files
    }), 200

if __name__ == '__main__':

    # Load environment variables
    load_dotenv()

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Load model
    load_models() 

    # Run app
    flask_app.run(host='0.0.0.0', port=8000)

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from dotenv import load_dotenv
import os
from groq import Groq
import util as ut

load_dotenv()

# AI model

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ML models

def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)
    
xgboost_model = load_model("../part-1/models/xgb_model.pkl")

naive_bayes_model = load_model("../part-1/models/nb_model.pkl")

random_forest_model = load_model("../part-1/models/rf_model.pkl")

decision_tree_model = load_model("../part-1/models/dt_model.pkl")

svm_model = load_model("../part-1/models/svm_model.pkl")

knn_model = load_model("../part-1/models/knn_model.pkl")

voting_classifier_model = load_model("../part-1/models/voting_clf_hard.pkl")

xgboost_SMOTE_model = load_model("../part-1/models/xgb_smote_model.pkl")

xgboost_feature_engineered_model = load_model("../part-1/models/xgb_feature_engineered_model.pkl")

def prepare_input(credit_score, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary, location, gender):

    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Gender_Female": 1 if gender == "Female" else 0
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict
    
def make_predictions(input_df):

    probabilities = {
        "XGBoost": xgboost_model.predict_proba(input_df)[0][1],
        "Random Forest": random_forest_model.predict_proba(input_df)[0][1],
        "K-Nearest Neighbors": knn_model.predict_proba(input_df)[0][1],
    }

    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {avg_probability:.2%} probability of churning.")

    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    return avg_probability

def get_percentiles(df, input_dict, columns):

    df_stats = df.describe()

    return { column: input_dict[column] / df_stats[column]["max"] if column != "CreditScore" else (input_dict[column] - 300) / (850 - 300) for column in columns }

def explain_prediction(probability, input_dict, surname):

    system_prompt = f"""You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.
    """

    prompt = f"""Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

    If the customer has less than a 30% risk of churning, generate a 3 sentence explanation of why they are at low risk of churning.

    If the customer has between 30% and 60% risk of churning, generate a 3 sentence explanation of why they are at moderate risk of churning.

    If the customer has over 60% risk of churning, generate a 3 sentence explanation of why they are at high risk of churning.
    
    Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.
 
    Respond based on the information provided below:

    Here is the customer's information:
    {input_dict}

    Here are the machine learning model's top 10 nost important features for predicting churn:

    Feature             | Importance
    --------------------------------
    NumOfProducts       | 0.323888
    IsActiveMember      | 0.164146
    Age                 | 0.109550
    Geography_Germany   | 0.091373
    Balance             | 0.052786
    Geography_France    | 0.046463
    Gender_Female       | 0.045283
    Geography_Spain     | 0.036855
    CreditScore         | 0.035005
    EstimatedSalary     | 0.032655
    HasCrCard           | 0.031940
    Tenure              | 0.030054
    Gender_Male         | 0.000000

    {pd.set_option("display.max_columns", None)}

    Here are summary statistics for churned customers:
    {df[df["Exited"] == 1].describe()}

    Here are summary statistics for non-churned customers:
    {df[df["Exited"] == 0].describe()}

    Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine ñearning model's prediction and top 10 most important features", just explain the prediction.
    """

    print("EXPLANATION PROMPT", prompt)

    raw_response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile", # llama-3.1-70b-versatile
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation, surname):

    system_prompt = f"""You are a manager at HS Bank. You are responsible for ensuring customers stay with the bank and are incentivized with various offers.
    """

    prompt = f"""You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

    Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that thet become more loyal to the bank.

    Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer.

    Here is the customer's information:
    {input_dict}

    Here is some explanation as to why the customer might be at risk of churning:
    {explanation}
    """

    print("EMAIL PROMPT", prompt)

    raw_response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile", # llama-3.1-70b-versatile
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    return raw_response.choices[0].message.content

# Frontend

st.title("Customer Churn Prediction")

df = pd.read_csv("../churn.csv")

customers = [f"{row["CustomerId"]} - {row["Surname"]}" for _, row in df.iterrows()]

seleted_customer_option = st.selectbox("Select a customer", customers)

if seleted_customer_option:
    selected_customer_id = int(seleted_customer_option.split(" - ")[0])
    selected_surname = seleted_customer_option.split(" - ")[1]

    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

col1, col2 = st.columns(2)

with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=int(selected_customer["CreditScore"])
    )

    location = st.selectbox(
        "Location", 
        ["Spain", "France", "Germany"],
        index=["Spain", "France", "Germany"].index(
            selected_customer["Geography"]
        )
    )

    gender = st.radio(
        "Gender",
        ["Male", "Female"],
        index=0 if selected_customer["Gender"] == "Male" else 1
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=int(selected_customer["Age"])
    )

    tenure = st.number_input(
        "Tenure (years)",
        min_value=0,
        max_value=50,
        value=int(selected_customer["Tenure"])
    )

with col2:

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=float(selected_customer["Balance"])
    )

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=10,
        value=int(selected_customer["NumOfProducts"])
    )

    has_credit_card = st.checkbox(
        "Has Credit Card",
        value=bool(selected_customer["HasCrCard"])
    )

    is_active_member = st.checkbox(
        "Is Active Member",
        value=bool(selected_customer["IsActiveMember"])
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=float(selected_customer["EstimatedSalary"])
    )

input_df, input_dict = prepare_input(credit_score, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary, location, gender)
avg_probability = make_predictions(input_df)

percentiles = get_percentiles(df, input_dict, ["NumOfProducts", "Balance", "EstimatedSalary", "Tenure", "CreditScore"])

fig = ut.create_customer_percentile_chart(percentiles)
st.plotly_chart(fig, use_container_width=True)

explanation = explain_prediction(avg_probability, input_dict, selected_customer["Surname"])

st.markdown("---")

st.subheader("Explanation of Prediction")

st.markdown(explanation)

email = generate_email(avg_probability, input_dict, explanation, selected_customer["Surname"])

st.markdown("---")

st.subheader("Personalized Email")

st.markdown(email)

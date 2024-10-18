# Overview

This project is divided into two branches: main and deployment.

## main

The main branch contains two parts. 

Part 1 is dedicated to exploring the data inside churn.csv and training different ML models. Furthermore, we used methods such as feature engineering, SMOTE, ensamble, and hyperparameter optimization to increase the performance (accuracy and recall) of the models.

Part 2 is dedicated to building a frontend, using the streamlit python library, to interact with the dataset and the trained models. More specifically, the user can visualize the probability of churn for a given customer based on the features and the trained models. Additionally, we use Groq to create LLM-powered responses to the reason about the churn probability and write personalized customer-retention email for the at risk customers. 

## deployment

The deployment branch contains a fully functional flask app that serves the frontend through API endpoints. These endpoints include loading the data, predicting the churn probability, and generating the LLM-powered responses. We deployed the app as an Azure Web App.

# Demo

You can find the demo of the project [here](https://customer-churn-prediction-101.streamlit.app/).

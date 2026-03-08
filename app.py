import streamlit as st
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator

st.title("Medical Diagnosis System using Bayesian Network")

# Load dataset
data = pd.read_csv("datasets/medical_data.csv")

# Define model structure
model = DiscreteBayesianNetwork([
    ('Fever', 'Disease'),
    ('Cough', 'Disease'),
    ('Fatigue', 'Disease')
])

# Train the model
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Inference
inference = VariableElimination(model)

st.header("Enter Patient Symptoms")

fever = st.selectbox("Fever", ["Yes", "No"])
cough = st.selectbox("Cough", ["Yes", "No"])
fatigue = st.selectbox("Fatigue", ["Yes", "No"])

if st.button("Predict Disease"):

    evidence = {
        'Fever': 1 if fever == "Yes" else 0,
        'Cough': 1 if cough == "Yes" else 0,
        'Fatigue': 1 if fatigue == "Yes" else 0
    }

    result = inference.query(variables=['Disease'], evidence=evidence)

    st.subheader("Prediction Result")
    st.write(result)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.metrics import confusion_matrix, accuracy_score

# Page configuration
st.set_page_config(
    page_title="Medical Diagnosis System",
    page_icon="🩺",
    layout="wide"
)

# Title
st.markdown(
    "<h1 style='text-align:center;'>🩺 Medical Diagnosis using Bayesian Network</h1>",
    unsafe_allow_html=True
)

st.divider()

# Load dataset
data = pd.read_csv("datasets/medical_data.csv")

# Define model
model = DiscreteBayesianNetwork([
    ('Fever', 'Disease'),
    ('Cough', 'Disease'),
    ('Fatigue', 'Disease')
])

# Train model
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Inference engine
inference = VariableElimination(model)

# -------------------------------
# Sidebar Inputs
# -------------------------------

st.sidebar.header("Patient Symptoms")

fever = st.sidebar.selectbox("Fever", ["Yes", "No"])
cough = st.sidebar.selectbox("Cough", ["Yes", "No"])
fatigue = st.sidebar.selectbox("Fatigue", ["Yes", "No"])

predict_button = st.sidebar.button("Predict Disease")

# -------------------------------
# Tabs Layout
# -------------------------------

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Evaluation", "Dataset"])

# -------------------------------
# Prediction Tab
# -------------------------------

with tab1:

    st.header("Disease Prediction")

    if predict_button:

        evidence = {
            'Fever': 1 if fever == "Yes" else 0,
            'Cough': 1 if cough == "Yes" else 0,
            'Fatigue': 1 if fatigue == "Yes" else 0
        }

        with st.spinner("Analyzing symptoms..."):
            result = inference.query(variables=['Disease'], evidence=evidence)

        probs = result.values
        labels = result.state_names['Disease']

        predicted = labels[probs.argmax()]

        st.success(f"Most Likely Diagnosis: {predicted}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Prediction Confidence", f"{max(probs):.2f}")

        with col2:
            st.metric("Number of Disease Classes", len(labels))

        st.subheader("Disease Probability Distribution")

        fig = px.bar(
            x=labels,
            y=probs,
            labels={'x': 'Disease', 'y': 'Probability'},
            title="Prediction Probability"
        )

        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Model Evaluation
# -------------------------------

with tab2:

    st.header("Model Performance")

    y_true = data['Disease']
    y_pred = []

    for _, row in data.iterrows():

        evidence = {
            'Fever': row['Fever'],
            'Cough': row['Cough'],
            'Fatigue': row['Fatigue']
        }

        prediction = inference.map_query(
            variables=['Disease'],
            evidence=evidence
        )

        y_pred.append(prediction['Disease'])

    accuracy = accuracy_score(y_true, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2f}")

    with col2:
        st.metric("Dataset Size", len(data))

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig2, ax2 = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        ax=ax2
    )

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix")

    st.pyplot(fig2)

# -------------------------------
# Dataset Tab
# -------------------------------

with tab3:

    st.header("Dataset Preview")

    st.dataframe(data)

    st.subheader("Feature Distribution")

    fig3 = px.histogram(
        data,
        x="Disease",
        title="Disease Distribution in Dataset"
    )

    st.plotly_chart(fig3, use_container_width=True)
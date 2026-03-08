import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.metrics import confusion_matrix, accuracy_score

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="AI Medical Diagnosis",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------

st.markdown("""
<style>

body {
    background-color: #f5f7fb;
}

.main-title {
    font-size:40px;
    font-weight:700;
    text-align:center;
    color:white;
}

.subtitle {
    text-align:center;
    color:white;
}

.header-box {
    background: linear-gradient(90deg,#0f2027,#203a43,#2c5364);
    padding:30px;
    border-radius:15px;
}

.metric-card {
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------

st.markdown("""
<div class="header-box">
<h1 class="main-title">🩺 AI Medical Diagnosis System</h1>
<p class="subtitle">
Bayesian Network powered disease prediction based on patient symptoms
</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ----------------------------
# LOAD DATA
# ----------------------------

data = pd.read_csv("datasets/medical_data.csv")

# ----------------------------
# MODEL
# ----------------------------

model = DiscreteBayesianNetwork([
    ('Fever', 'Disease'),
    ('Cough', 'Disease'),
    ('Fatigue', 'Disease')
])

model.fit(data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.title("🧑‍⚕️ Patient Symptoms")

st.sidebar.markdown("Enter symptoms to predict disease")

fever = st.sidebar.selectbox("🌡 Fever", ["Yes", "No"])
cough = st.sidebar.selectbox("😷 Cough", ["Yes", "No"])
fatigue = st.sidebar.selectbox("😴 Fatigue", ["Yes", "No"])

predict_button = st.sidebar.button("🔍 Predict Disease")

st.sidebar.markdown("---")

st.sidebar.info("AI model trained using Bayesian Networks")

# ----------------------------
# KPI DASHBOARD
# ----------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Dataset Size", len(data))
col2.metric("Model Type", "Bayesian Network")
col3.metric("Features", 3)

st.divider()

# ----------------------------
# TABS
# ----------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Prediction",
    "📊 Model Evaluation",
    "📂 Dataset",
    "🧬 Network Graph"
])

# ----------------------------
# PREDICTION TAB
# ----------------------------

with tab1:

    st.subheader("Disease Prediction")

    if predict_button:

        evidence = {
            'Fever': 1 if fever == "Yes" else 0,
            'Cough': 1 if cough == "Yes" else 0,
            'Fatigue': 1 if fatigue == "Yes" else 0
        }

        with st.spinner("🧠 AI analyzing symptoms..."):
            result = inference.query(
                variables=['Disease'],
                evidence=evidence
            )

        probs = result.values
        labels = result.state_names['Disease']

        predicted = labels[probs.argmax()]

        st.success(f"Predicted Disease: **{predicted}**")

        # Gauge confidence
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=max(probs)*100,
            title={'text': "Prediction Confidence"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {'color': "#2c5364"},
                'steps':[
                    {'range':[0,50],'color':'#ffcccc'},
                    {'range':[50,80],'color':'#ffe680'},
                    {'range':[80,100],'color':'#ccffcc'}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Disease Probability Distribution")

        fig2 = px.bar(
            x=labels,
            y=probs,
            color=labels,
            labels={'x':'Disease','y':'Probability'}
        )

        st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# MODEL EVALUATION
# ----------------------------

with tab2:

    st.subheader("Model Performance")

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

    st.metric("Model Accuracy", f"{accuracy:.2f}")

    cm = confusion_matrix(y_true, y_pred)

    fig3, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig3)

# ----------------------------
# DATASET TAB
# ----------------------------

with tab3:

    st.subheader("Dataset Preview")

    st.dataframe(data)

    col1, col2 = st.columns(2)

    with col1:

        fig4 = px.pie(
            data,
            names="Disease",
            title="Disease Distribution"
        )

        st.plotly_chart(fig4)

    with col2:

        fig5 = px.histogram(
            data,
            x="Fever",
            color="Disease",
            title="Fever vs Disease"
        )

        st.plotly_chart(fig5)

# ----------------------------
# NETWORK GRAPH
# ----------------------------

with tab4:

    st.subheader("Bayesian Network Structure")

    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    fig6, ax = plt.subplots()

    nx.draw(
        G,
        with_labels=True,
        node_color="#9fd3ff",
        node_size=3000,
        font_size=12,
        ax=ax
    )

    st.pyplot(fig6)
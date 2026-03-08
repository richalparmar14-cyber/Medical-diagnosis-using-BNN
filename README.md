# 🏥 Medical Diagnosis using Bayesian Networks

## 📌 Project Overview

This project implements a **Bayesian Network-based Medical Diagnosis System** that predicts diseases based on symptoms and risk factors using probabilistic reasoning.

Medical diagnosis involves uncertainty. Traditional rule-based systems cannot effectively handle incomplete or uncertain data. This system uses **Bayes’ Theorem** and probabilistic graphical models to compute posterior probabilities of diseases.

---

## 🧠 Concept Used

Bayesian Networks were introduced by Judea Pearl and are based on Bayes' Theorem:

P(D|S) = (P(S|D) × P(D)) / P(S)

Where:
- P(D|S) → Posterior probability (Disease given Symptoms)
- P(S|D) → Likelihood
- P(D) → Prior probability
- P(S) → Evidence probability

---

## 🏥 Diseases Considered

- Flu
- COVID-19
- Malaria

---

## 🤒 Symptoms Used

- Fever
- Cough
- Headache
- Fatigue

---

## ⚠ Risk Factors

- Travel History
- Mosquito Exposure

---

## 🛠 Technologies Used

- Python
- pgmpy (Probabilistic Graphical Models)
- pandas
- numpy
- Git & GitHub

---

## 📂 Project Structure
Medical-Diagnosis-Bayesian-Network/
│
├── dataset/
│ └── medical_data.csv
│
├── src/
│ ├── model.py
│ ├── train_model.py
│ ├── inference.py
│
├── main.py
├── requirements.txt
└── README.md

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
pip install -r requirements.txt


### 2️⃣ Run the Program


python main.py


---

## 📊 Sample Output


Diagnosis Result:

COVID : 0.42
Flu : 0.35
Malaria : 0.23


The system calculates posterior probabilities dynamically based on input symptoms.

---

## 🔬 Applications

- Clinical Decision Support Systems
- Disease Risk Prediction
- AI-based Healthcare Systems
- Uncertainty Modeling in Medicine

---

## 📈 Advantages of Bayesian Networks

✔ Handles uncertainty  
✔ Works with incomplete data  
✔ Provides probability-based output  
✔ Transparent reasoning process  

---

## 🎓 Academic Relevance

This project demonstrates the practical implementation of:

- Probabilistic Graphical Models
- Bayesian Inference
- Machine Learning in Healthcare
- Medical AI Systems

---

## 👩‍💻 Author

Richal  
B.Tech Student AIMLD  
Mini Project – Medical Diagnosis using Bayesian Networks

---

## 📜 License

This project is developed for academic and educational purposes.


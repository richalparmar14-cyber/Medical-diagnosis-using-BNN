import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from pgmpy.inference import VariableElimination
from .train_model import train


def evaluate_model():
    # Get dataset path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "datasets", "medical_data.csv")

    # Load dataset
    data = pd.read_csv(data_path)

    # Train model
    model = train()

    # Create inference object
    inference = VariableElimination(model)

    y_true = []
    y_pred = []

    # Iterate through dataset
    for index, row in data.iterrows():
        evidence = {
            'Fever': row['Fever'],
            'Cough': row['Cough'],
            'Headache': row['Headache'],
            'Fatigue': row['Fatigue'],
            'Travel_History': row['Travel_History'],
            'Mosquito_Exposure': row['Mosquito_Exposure']
        }

        # Predict disease
        result = inference.query(variables=['Disease'], evidence=evidence)

        predicted = result.values.argmax()
        predicted_label = result.state_names['Disease'][predicted]

        y_true.append(row['Disease'])
        y_pred.append(predicted_label)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title("Confusion Matrix - Medical Diagnosis BNN")
    plt.show()


if __name__ == "__main__":
    evaluate_model()
import os
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from .model import create_model

def train():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "datasets", "medical_data.csv")
    data = pd.read_csv(data_path)

    model = create_model()
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model

if __name__ == "__main__":
    train()
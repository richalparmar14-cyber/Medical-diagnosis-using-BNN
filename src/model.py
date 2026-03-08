from pgmpy.models import DiscreteBayesianNetwork

def create_model():
    model = DiscreteBayesianNetwork([
        ('Travel_History', 'Disease'),
        ('Mosquito_Exposure', 'Disease'),
        ('Disease', 'Fever'),
        ('Disease', 'Cough'),
        ('Disease', 'Headache'),
        ('Disease', 'Fatigue')
    ])
    return model
import joblib
import numpy as np


def evaluate_model():
    model = joblib.load("Models/model.pkl")
    test_x = np.array([[6],[7],[8]])
    predictions = model.predict(test_x)
    
    print("Prediction",predictions)
    
if __name__ == "__main__":
    evaluate_model()
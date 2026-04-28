from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Load PHQ-9 (SVM) and Scaler
phq9_model = joblib.load('svm_phq9_model.pkl')
phq9_scaler = joblib.load('scaler_phq9.pk1')

# Load BRS (CNN)
brs_model = load_model('brs_cnn_with_demographics.keras')

class InputData(BaseModel):
    values: list # A list of integers (answers)

@app.post("/predict_phq9")
def predict_phq9(data: InputData):
    # Data order: phq9_1 to phq9_9, sex, age_group [cite: 60]
    scaled = phq9_scaler.transform([data.values])
    prediction = phq9_model.predict(scaled)
    return {"result": int(prediction[0])}

@app.post("/predict_brs")
def predict_brs(data: InputData):
    # Data order: BRS_1 to BRS_6, sex, age_group [cite: 163]
    # Reshape to 3D for the CNN: (1, 8, 1) [cite: 174, 182]
    array_data = np.array(data.values).reshape(1, 8, 1)
    prediction = brs_model.predict(array_data)
    class_idx = np.argmax(prediction, axis=1)
    return {"result": int(class_idx[0])}
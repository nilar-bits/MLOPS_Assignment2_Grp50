from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the saved model and scaler
model, scaler = joblib.load('linear_svc_model.pkl')


# Define the input data model
class DataPoint(BaseModel):
    input_features: str


# Endpoint for prediction
@app.post("/predict/")
async def predict(data: DataPoint):
    try:
        # Step 1: Convert the string into a numpy array
        single_data_point = np.array(data.input_features.split(','), dtype=float).reshape(1, -1)

        # Step 2: Scale the data point
        single_data_point_scaled = scaler.transform(single_data_point)

        # Step 3: Use the model to make a prediction
        prediction = model.predict(single_data_point_scaled)

        # Step 4: Return the result as a response
        result = "Benign" if prediction[0] == 1 else "Malignant"
        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}

# Run the FastAPI application using `uvicorn` if running locally
# Example: uvicorn main:app --reload

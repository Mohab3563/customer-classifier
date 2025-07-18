from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.cluster import KMeans
import joblib

# Load trained KMeans model
model = joblib.load("app/model/kmeans.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define input schema
class CustomerData(BaseModel):
    age: int
    annual_income: float
    spending_score: float

# Define mapping of clusters to richness levels
richness_map = {
    0: ("C Class", "Low Richness"),
    1: ("B Class", "Moderate Richness"),
    2: ("A Class", "High Richness")
}

@app.get("/")
def root():
    return {"message": "KMeans Customer Classification API is running 🚀"}

@app.post("/predict")
def predict(data: CustomerData):
    X_new = [[data.age, data.annual_income, data.spending_score]]
    cluster = model.predict(X_new)[0]

    # Get class label and description
    class_label, description = richness_map.get(cluster, ("Unknown Class", "Unknown Richness"))

    return {
        "class": class_label,
        "message": f"This customer belongs to {class_label} ({description})"
    }

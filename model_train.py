import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Define column names
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

# Load data using relative path
try:
    data = pd.read_csv("processed.cleveland.csv", names=column_names, na_values='?')
    data = data.dropna()
    
    # Train model
    X = data.drop('target', axis=1)
    y = data['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model using relative path
    with open('heartdisease1.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved successfully!")
    
    # Optional: Print model accuracy
    accuracy = model.score(X, y)
    print(f"Model accuracy: {accuracy:.4f}")

except Exception as e:
    print(f"An error occurred: {e}")

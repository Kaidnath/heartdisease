import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
<<<<<<< HEAD

# Load data with full path
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
data = pd.read_csv("/Users/zoom/Desktop/HeartDisease/processed.cleveland.csv", names=column_names, na_values='?')
data = data.dropna()

# Train model
X = data.drop('target', axis=1)
y = data['target']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model with full path
with open('/Users/zoom/Desktop/HeartDisease/heartdisease.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved!")
=======
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
>>>>>>> 5ab09b7af41f214a06d6ffaa954eb64fbb32151a

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

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
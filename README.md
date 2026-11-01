# Student_performance_predictor

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# 1. Load the dataset[cite: 304]
data = pd.read_csv('data.csv')

# 2. Separate Features (Inputs) and Target[cite: 304]
X = data[['study_hours', 'attendance', 'previous_score']]
y = data['final_score']

# 3. Split data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model[cite: 334]
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 5. Save the trained model using joblib
joblib.dump(model, 'model.pkl')
print("Model Trained Successfully and saved as model.pkl!")

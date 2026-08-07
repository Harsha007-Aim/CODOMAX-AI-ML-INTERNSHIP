import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Dataset
data = {
    'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7,
              7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4,
              2.7, 4.8, 3.8, 6.9, 7.8],
    'Scores': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25,
               85, 62, 41, 42, 17, 95, 30, 24, 67, 69,
               30, 54, 35, 76, 86]
}
df = pd.DataFrame(data)

print("=" * 50)
print("  Student Score Prediction System")
print("  Codomax AI & ML Internship")
print("=" * 50)

# 2. Prepare Data
X = df[['Hours']]
y = df['Scores']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")
print(f"Coefficient : {model.coef_[0]:.4f}")
print(f"Intercept   : {model.intercept_:.4f}")

# 4. Predictions
y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
result = pd.DataFrame({
    'Hours': X_test.values.flatten(),
    'Actual': y_test.values,
    'Predicted': np.round(y_pred, 2)
})
print(result.to_string(index=False))

# 5. Evaluation
print("\nModel Evaluation:")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE  : {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2   : {r2_score(y_test, y_pred):.4f}")

# 6. Final Prediction Example
hours = 9.25
score = model.predict([[hours]])[0]
print(f"\nPredicted Score for {hours} hours: {score:.2f}")

print("\nProject Completed Successfully!")
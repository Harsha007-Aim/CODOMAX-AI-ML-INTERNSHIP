import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------
# 1. Create the Dataset (No internet required)
# ------------------------------------------------------
data = {
    'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 
              7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 
              2.7, 4.8, 3.8, 6.9, 7.8],
    'Scores': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 
               85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 
               30, 54, 35, 76, 86]
}

df = pd.DataFrame(data)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)
print(df.head())

# ------------------------------------------------------
# 2. Separate Features and Target
# ------------------------------------------------------
X = df[['Hours']]
y = df['Scores']

# ------------------------------------------------------
# 3. Train-Test Split
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n" + "=" * 60)
print("DATA SPLIT COMPLETED")
print("=" * 60)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")

# ------------------------------------------------------
# 4. Train the Linear Regression Model
# ------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("MODEL TRAINED SUCCESSFULLY")
print("=" * 60)

# ------------------------------------------------------
# 5. Make Predictions
# ------------------------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------------------------
# 6. Model Evaluation using MAE, MSE, and R² Score
# ------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"Mean Absolute Error (MAE)      : {mae:.2f}")
print(f"Mean Squared Error (MSE)       : {mse:.2f}")
print(f"Root Mean Squared Error (RMSE) : {rmse:.2f}")
print(f"R² Score                       : {r2:.4f}")

# ------------------------------------------------------
# 7. Explanation of Metrics
# ------------------------------------------------------
print("\n" + "=" * 60)
print("WHAT DO THESE METRICS MEAN?")
print("=" * 60)
print("""
→ MAE  : Average absolute difference between actual and predicted scores.
→ MSE  : Average of squared differences (penalizes larger errors more).
→ RMSE : Square root of MSE (in the same unit as scores).
→ R²   : How well the model explains the variance in scores.
         (Closer to 1.0 means better model performance)
""")

# ------------------------------------------------------
# 8. Actual vs Predicted Comparison
# ------------------------------------------------------
print("=" * 60)
print("ACTUAL vs PREDICTED SCORES")
print("=" * 60)

comparison = pd.DataFrame({
    'Study Hours'     : X_test.values.flatten(),
    'Actual Score'    : y_test.values,
    'Predicted Score' : np.round(y_pred, 2)
})

print(comparison.to_string(index=False))

print("\n" + "=" * 60)
print("DAY 10 COMPLETED SUCCESSFULLY!")
print("Model performance has been measured using MAE, MSE and R² Score.")
print("=" * 60)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
print(f"Coefficient (m) : {model.coef_[0]:.4f}")
print(f"Intercept (c)   : {model.intercept_:.4f}")

# ------------------------------------------------------
# 5. Make Predictions on Test Data
# ------------------------------------------------------
y_pred = model.predict(X_test)

print("\n" + "=" * 60)
print("PREDICTIONS ON TEST DATA")
print("=" * 60)

comparison = pd.DataFrame({
    'Study Hours': X_test.values.flatten(),
    'Actual Score': y_test.values,
    'Predicted Score': np.round(y_pred, 2)
})

print(comparison.to_string(index=False))

# ------------------------------------------------------
# 6. Predict Score for Custom Study Hours
# ------------------------------------------------------
print("\n" + "=" * 60)
print("CUSTOM PREDICTIONS")
print("=" * 60)

# Example 1: Classic question
hours1 = 9.25
score1 = model.predict([[hours1]])
print(f"If a student studies for {hours1} hours → Predicted Score: {score1[0]:.2f}")

# Example 2
hours2 = 5.5
score2 = model.predict([[hours2]])
print(f"If a student studies for {hours2} hours → Predicted Score: {score2[0]:.2f}")

# Example 3
hours3 = 7.0
score3 = model.predict([[hours3]])
print(f"If a student studies for {hours3} hours → Predicted Score: {score3[0]:.2f}")

# Example 4
hours4 = 3.5
score4 = model.predict([[hours4]])
print(f"If a student studies for {hours4} hours → Predicted Score: {score4[0]:.2f}")

print("\n" + "=" * 60)
print("DAY 9 COMPLETED SUCCESSFULLY!")
print("Predictions generated successfully using the trained model.")
print("=" * 60)
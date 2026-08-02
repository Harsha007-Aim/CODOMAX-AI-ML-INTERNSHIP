# ======================================================
# Student Score Prediction System
# Codomax AI & ML Internship
# ======================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------
# 1. Create the Dataset (No internet required)
# ------------------------------------------------------
data = {
    'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.8],
    'Scores': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 30, 54, 35, 76, 86]
}

df = pd.DataFrame(data)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)
print(df.head())
print("\nShape of dataset:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------------------------------------------
# 2. Understanding Supervised Learning
# ------------------------------------------------------
print("\n" + "=" * 60)
print("WHAT IS SUPERVISED LEARNING?")
print("=" * 60)
print("""
Supervised Learning means the model learns from labeled data.
We already know the correct answers (Scores) for given inputs (Hours).

In this project:
→ Input (Feature)  : Hours studied
→ Output (Target)  : Exam Scores

The model learns the relationship between Hours and Scores.
""")

# ------------------------------------------------------
# 3. Separate Features (X) and Target (y)
# ------------------------------------------------------
X = df[['Hours']]
y = df['Scores']

print("=" * 60)
print("FEATURES (X) AND TARGET (y)")
print("=" * 60)
print("\nFeatures (X) - First 5 rows:")
print(X.head())
print("\nTarget (y) - First 5 values:")
print(y.head())

# ------------------------------------------------------
# 4. Train-Test Split
# ------------------------------------------------------
print("\n" + "=" * 60)
print("TRAIN - TEST SPLIT")
print("=" * 60)
print("""
We divide the data into two parts:
1. Training Set (80%) → Used to train the model
2. Testing Set  (20%) → Used to test the model performance
""")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nAfter Split:")
print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"y_test shape  : {y_test.shape}")

print("\nTraining data (first 5 rows):")
print(X_train.head())

# ------------------------------------------------------
# 5. Linear Regression Concept
# ------------------------------------------------------
print("\n" + "=" * 60)
print("LINEAR REGRESSION CONCEPT")
print("=" * 60)
print("""
Linear Regression finds the best straight line that fits the data.

Equation:   y = m * x + c

Where:
→ y = Predicted Score
→ x = Hours studied
→ m = Slope (Coefficient)
→ c = Intercept
""")

# ------------------------------------------------------
# 6. Train the Model
# ------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("=" * 60)
print("MODEL TRAINED SUCCESSFULLY")
print("=" * 60)

print(f"\nCoefficient (m) : {model.coef_[0]:.4f}")
print(f"Intercept (c)   : {model.intercept_:.4f}")

print("\nInterpretation:")
print(f"→ For every 1 extra hour of study, score increases by approximately {model.coef_[0]:.2f} marks.")
print(f"→ When study hours = 0, the base score is approximately {model.intercept_:.2f}")

# ------------------------------------------------------
# 7. Simple Prediction
# ------------------------------------------------------
print("\n" + "=" * 60)
print("SIMPLE PREDICTION EXAMPLE")
print("=" * 60)

sample_hours = 7.5
predicted_score = model.predict([[sample_hours]])

print(f"If a student studies for {sample_hours} hours,")
print(f"Predicted Score ≈ {predicted_score[0]:.2f}")

print("\n" + "=" * 60)
print("DAY 7 COMPLETED SUCCESSFULLY!")
print("=" * 60)
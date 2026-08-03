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
print("\nTotal records:", df.shape[0])

# ------------------------------------------------------
# 2. Separate Features (X) and Target (y)
# ------------------------------------------------------
X = df[['Hours']]     # Feature (2D)
y = df['Scores']      # Target (1D)

print("\n" + "=" * 60)
print("FEATURES AND TARGET READY")
print("=" * 60)
print("Feature (X) shape:", X.shape)
print("Target (y) shape :", y.shape)

# ------------------------------------------------------
# 3. Train-Test Split
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print("\n" + "=" * 60)
print("DATA SPLIT COMPLETED")
print("=" * 60)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")

# ------------------------------------------------------
# 4. Create Linear Regression Model
# ------------------------------------------------------
model = LinearRegression()

print("\n" + "=" * 60)
print("LINEAR REGRESSION MODEL CREATED")
print("=" * 60)
print("Model Object:", model)

# ------------------------------------------------------
# 5. Train the Model
# ------------------------------------------------------
model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("MODEL TRAINED SUCCESSFULLY!")
print("=" * 60)

# ------------------------------------------------------
# 6. Display Model Parameters
# ------------------------------------------------------
print("\nModel Parameters:")
print(f"Coefficient (m) : {model.coef_[0]:.4f}")
print(f"Intercept (c)   : {model.intercept_:.4f}")

print("\nEquation of the line:")
print(f"Score = {model.coef_[0]:.4f} × Hours + {model.intercept_:.4f}")

# ------------------------------------------------------
# 7. Make Predictions on Test Data (for verification)
# ------------------------------------------------------
y_pred = model.predict(X_test)

print("\n" + "=" * 60)
print("PREDICTIONS ON TEST DATA")
print("=" * 60)

comparison = pd.DataFrame({
    'Actual Score': y_test.values,
    'Predicted Score': y_pred.round(2)
})
print(comparison)

print("\n" + "=" * 60)
print("DAY 8 COMPLETED SUCCESSFULLY!")
print("Linear Regression model has been created and trained.")
print("=" * 60)
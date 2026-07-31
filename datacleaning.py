# -------------------------------------------------
# Data Cleaning Tasks
# 1. Handle missing values
# 2. Remove duplicates
# 3. Understand dataset statistics
# -------------------------------------------------

import pandas as pd
import numpy as np

# -------------------------------------------------
# Create a sample student score dataset
# (Intentionally added missing values & duplicates)
# -------------------------------------------------
data = {
    "Student_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                   102, 111, 112],          # 102 appears twice (duplicate)
    "Name": ["Aarav", "Priya", "Rohan", "Sneha", "Vikram",
             "Ananya", "Karan", "Meera", "Arjun", "Isha",
             "Priya", "Rahul", "Neha"],
    "Math": [85, 92, np.nan, 88, 95, 70, 82, 91, np.nan, 89, 92, 77, 84],
    "Science": [90, 85, 88, np.nan, 87, 75, 80, 94, 79, 86, 85, 81, np.nan],
    "English": [78, 88, 82, 85, 90, np.nan, 75, 89, 81, 84, 88, 79, 86],
    "Total": [253, 265, 248, 265, 272, 213, 237, 274, 236, 259, 265, 237, 250]
}

df = pd.DataFrame(data)

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(df)
print(f"\nShape: {df.shape}")
print()

# -------------------------------------------------
# 1. Handle Missing Values
# -------------------------------------------------
print("=" * 60)
print("1. HANDLING MISSING VALUES")
print("=" * 60)

# Check how many missing values are present
print("\nMissing values before cleaning:")
print(df.isnull().sum())
print()

# Option A: Fill numerical columns with mean
df["Math"] = df["Math"].fillna(df["Math"].mean())
df["Science"] = df["Science"].fillna(df["Science"].mean())
df["English"] = df["English"].fillna(df["English"].mean())

# Recalculate Total after filling (optional but good practice)
df["Total"] = df[["Math", "Science", "English"]].sum(axis=1).round(0)

print("Missing values after filling with mean:")
print(df.isnull().sum())
print()

# -------------------------------------------------
# 2. Remove Duplicates
# -------------------------------------------------
print("=" * 60)
print("2. REMOVING DUPLICATES")
print("=" * 60)

print(f"Number of duplicate rows before: {df.duplicated().sum()}")

# Remove complete duplicate rows
df = df.drop_duplicates()

# (Optional) Remove duplicates based on a specific column (e.g. Student_ID)
# df = df.drop_duplicates(subset=["Student_ID"], keep="first")

print(f"Number of duplicate rows after : {df.duplicated().sum()}")
print(f"Shape after removing duplicates: {df.shape}")
print()

# -------------------------------------------------
# 3. Understand Dataset Statistics
# -------------------------------------------------
print("=" * 60)
print("3. DATASET STATISTICS")
print("=" * 60)

print("\n--- Basic Info ---")
df.info()
print()

print("--- Statistical Summary (Numerical Columns) ---")
print(df.describe().round(2))
print()

print("--- Column Names ---")
print(df.columns.tolist())
print()

print("--- Final Cleaned Dataset ---")
print(df.reset_index(drop=True))
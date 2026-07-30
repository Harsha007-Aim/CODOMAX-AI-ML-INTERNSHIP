# -------------------------------------------------
# Pandas Task: Import, Load & Explore Student Scores
# -------------------------------------------------

# 1. Import Pandas
import pandas as pd

# 2. Create / Load the student score dataset
#    (Sample data written directly so the code is completely self-contained)
data = {
    "Student_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "Name": ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", 
             "Ananya", "Karan", "Meera", "Arjun", "Isha"],
    "Math": [85, 92, 78, 88, 95, 70, 82, 91, 76, 89],
    "Science": [90, 85, 88, 92, 87, 75, 80, 94, 79, 86],
    "English": [78, 88, 82, 85, 90, 68, 75, 89, 81, 84],
    "Total": [253, 265, 248, 265, 272, 213, 237, 274, 236, 259]
}

df = pd.DataFrame(data)

print("✅ Dataset loaded successfully!\n")

# 3. Explore the dataset
print("=" * 50)
print("1. First 5 rows of the dataset:")
print("=" * 50)
print(df.head())
print()

print("=" * 50)
print("2. Last 5 rows of the dataset:")
print("=" * 50)
print(df.tail())
print()

print("=" * 50)
print("3. Shape of the dataset (rows, columns):")
print("=" * 50)
print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")
print(f"Shape             : {df.shape}")
print()

print("=" * 50)
print("4. Column names:")
print("=" * 50)
print(df.columns.tolist())
print()

print("=" * 50)
print("5. Dataset information (dtypes, non-null counts, memory):")
print("=" * 50)
df.info()
print()

print("=" * 50)
print("6. Statistical summary of numerical columns:")
print("=" * 50)
print(df.describe())
print()

print("=" * 50)
print("7. Check for missing values:")
print("=" * 50)
print(df.isnull().sum())
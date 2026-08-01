
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------
# 1. Load the Dataset
# ----------------------------------------------
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)

print("Dataset loaded successfully!")
print(df.head())
print("\nShape of dataset:", df.shape)

# ----------------------------------------------
# 2. Scatter Plot (Hours vs Scores)
# ----------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df['Hours'], df['Scores'], color='blue', s=80, edgecolor='black', alpha=0.7)
plt.title('Scatter Plot: Study Hours vs Exam Scores', fontsize=16, fontweight='bold')
plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Exam Scores', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ----------------------------------------------
# 3. Bar Chart (Hours vs Scores)
# ----------------------------------------------
# Sorting for better visualization
df_sorted = df.sort_values(by='Hours')

plt.figure(figsize=(12, 6))
plt.bar(df_sorted['Hours'].astype(str), df_sorted['Scores'], color='skyblue', edgecolor='black')
plt.title('Bar Chart: Study Hours vs Exam Scores', fontsize=16, fontweight='bold')
plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Exam Scores', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ----------------------------------------------
# 4. Line Chart (Hours vs Scores)
# ----------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df_sorted['Hours'], df_sorted['Scores'], 
         color='green', marker='o', linestyle='-', linewidth=2, markersize=8)
plt.title('Line Chart: Study Hours vs Exam Scores', fontsize=16, fontweight='bold')
plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Exam Scores', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ----------------------------------------------
# 5. Combined View (Optional - All 3 in one figure)
# ----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scatter
axes[0].scatter(df['Hours'], df['Scores'], color='blue', s=70, edgecolor='black', alpha=0.7)
axes[0].set_title('Scatter Plot', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Hours Studied')
axes[0].set_ylabel('Scores')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Bar
axes[1].bar(df_sorted['Hours'].astype(str), df_sorted['Scores'], color='skyblue', edgecolor='black')
axes[1].set_title('Bar Chart', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Hours Studied')
axes[1].set_ylabel('Scores')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

# Line
axes[2].plot(df_sorted['Hours'], df_sorted['Scores'], color='green', marker='o', linewidth=2)
axes[2].set_title('Line Chart', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Hours Studied')
axes[2].set_ylabel('Scores')
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.suptitle('Data Visualization - Student Scores Dataset', fontsize=18, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()

print("\nAll charts created successfully!")
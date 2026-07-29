import numpy as np

# Create arrays
hours = np.array([2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7])
scores = np.array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25])

print("Hours array:", hours)
print("Shape:", hours.shape)
print("Mean hours:", np.mean(hours))
print("Max score:", np.max(scores))

# Indexing & Slicing
print(hours[0])
print(hours[2:5])

# Math operations
print(hours * 2)
print(np.sqrt(scores))
print(np.dot(hours, scores))   # if same length
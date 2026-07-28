# Variables & Data Types
name = "Student"
hours = 5.5
score = 75
is_passed = True

print(type(hours), type(score))

# Operators
print(10 + 5)
print(10 ** 2)   # power
print(15 % 4)    # remainder

# Loops
for i in range(1, 6):
    print(f"Study hour {i}")

# Function
def predict_score(hours):
    return hours * 9.5 + 2.5   # simple formula for practice

print(predict_score(7.5))
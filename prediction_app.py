# prediction_app.py
from sklearn.linear_model import LinearRegression
import pandas as pd

# Dataset included inside the code
data = {
    'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7,
              7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4,
              2.7, 4.8, 3.8, 6.9, 7.8],
    'Scores': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25,
               85, 62, 41, 42, 17, 95, 30, 24, 67, 69,
               30, 54, 35, 76, 86]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Scores']

model = LinearRegression()
model.fit(X, y)

print("Student Score Prediction App is ready!")
print("Enter study hours to get predicted score.")
print("Type -1 to exit.\n")

while True:
    try:
        hours = float(input("Enter study hours (or -1 to exit): "))
        
        if hours == -1:
            print("Exiting the app. Thank you!")
            break
            
        if hours < 0:
            print("Hours cannot be negative!")
            continue
            
        score = model.predict([[hours]])[0]
        print(f"Predicted Score: {score:.2f}\n")
        
    except ValueError:
        print("Please enter a valid number.\n")
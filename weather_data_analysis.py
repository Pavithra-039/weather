import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv("seattle-weather.csv")
print("Dataset Loaded Successfully")

print("\nSample Data:")
print(data.head())

data['date'] = pd.to_datetime(data['date'])

data['year'] = data['date'].dt.year

X = data[['precipitation', 'temp_min', 'wind']]
y = data['temp_max']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
svr = SVR(kernel='rbf')

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
svr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_svr = svr.predict(X_test)

results = pd.DataFrame({
    'Model': ['Linear Regression','Decision Tree','SVR'],
    'MAE': [
        mean_absolute_error(y_test,y_pred_lr),
        mean_absolute_error(y_test,y_pred_dt),
        mean_absolute_error(y_test,y_pred_svr)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test,y_pred_lr)),
        np.sqrt(mean_squared_error(y_test,y_pred_dt)),
        np.sqrt(mean_squared_error(y_test,y_pred_svr))
    ],
    'R2 Score': [
        r2_score(y_test,y_pred_lr),
        r2_score(y_test,y_pred_dt),
        r2_score(y_test,y_pred_svr)
    ]
})

print("\nModel Comparison Results:")
print(results)

plt.figure(figsize=(10,6))

plt.scatter(y_test, y_pred_lr, color='red', label='Linear Regression')
plt.scatter(y_test, y_pred_dt, color='green', label='Decision Tree')
plt.scatter(y_test, y_pred_svr, color='blue', label='SVR')

plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Weather Temperature Prediction Comparison")
plt.legend()
plt.grid(True)

plt.show()
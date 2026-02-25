# modules

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loading dataset

df = pd.read_csv("C:/Users/Tendulkar/Downloads/archive/salary_dataset.csv")

# print(df.head())
# converting into 2D array
X= np.array(df["YearsExperience"]).reshape(-1,1) # 2d(reshape(rows,columns))
y= np.array(df["Salary"]).reshape(-1,1)

# creating model for the graph
X_train, X_test, y_train, y_test = train_test_split(X, y) #passing data
model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# print(X.shape,y.shape)
#plotting graph
plt.scatter(X,y, c='r')
plt.plot(X, y, c='b')
plt.plot(X_test, y_predict, c="g", linewidth=3)
plt.show()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("C:/Users/Tendulkar/Downloads/archive/salary_dataset.csv")

# print(df.head())

X = df[["YearsExperience"]] #double brackets ensures column as a dataframe not series
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(r2_score(y_test, y_pred))
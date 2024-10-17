# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#import the data to development env.
data = pd.read_csv('selling_data.csv')

#preprocessing
months = data.iloc[:, 0:1]
sales = data.iloc[:, 1:2]

#set train and test data
x_train, x_test, y_train, y_test = train_test_split(months, sales, test_size = 0.33, random_state = 0)

#build the model
model = LinearRegression()
model.fit(x_train, y_train) #train the model by the data
prediction_result = model.predict(x_test) #obtain the prediction values of test data

#to accurate data matching, sort the indexes
x_train = x_train.sort_index()
y_train = y_train.sort_index()

#visualization training data and prediction result
plt.title('Linear Regression')
plt.xlabel('Month')
plt.ylabel('Sale Value')

plt.scatter(x_train, y_train) #training data
plt.plot(x_test, prediction_result, c='red') #the result of linear regression

#development section
concated = pd.concat([x_test, y_test], axis=1)
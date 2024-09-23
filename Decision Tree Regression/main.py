import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('salary.csv')
df = df.drop(['unvan'], axis=1) #dropped categorial row

#spliting rows to use on the training
education_level = pd.DataFrame(data=df.iloc[:, 0:1].values, index=range(10), columns=['education_level'])
salary = pd.DataFrame(data=df.iloc[:, 1:2].values, index=range(10), columns=['salary'])

#creating the model
regressor = DecisionTreeRegressor()
regressor.fit(education_level, salary)

#visualizating the prediction
plt.figure()
plt.scatter(education_level, salary, color='red')
plt.plot(education_level, regressor.predict(education_level), color='blue')
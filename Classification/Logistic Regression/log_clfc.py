import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# import the csv file which is contain the data
df = pd.read_csv('data.csv')
df = df.drop(['ulke'], axis=1) #dropped categorial row

data_input = df.iloc[:, 0:3]
data_output = df.iloc[:, -1:]

# splitting test and train data
x_train, x_test, y_train, y_test = train_test_split(data_input, data_output, test_size=0.33, random_state=0)

# standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)

# logistic regression model
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
prediction = logistic_reg.predict(X_test)

# print results as compared
print(f"""
Prediction result: 
{prediction}

The real output:
{y_test.values}
""")
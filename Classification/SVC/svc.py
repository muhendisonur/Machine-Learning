import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
classifier = SVC(kernel='rbf') # by decreasing neighbors number, model will be avoid outliner data(the dataset has outliner data). So, it will be predict better
classifier.fit(X_train, y_train)
result = classifier.predict(X_test)

# running confusion matrix
cm_result = confusion_matrix(y_test, result)

# print results as compared
print(f"""
Real Output:
{y_test.values}

Predictions:
{result}

Confusion Matrix Result:
{cm_result}
""")

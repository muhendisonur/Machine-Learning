import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

# import the dataset
data = pd.read_excel('Iris.xls')

# handle training data and result on different dataframes
iris = data.iloc[:, -1:]
data = data.drop(['iris'], axis=1)

# split train and test data from the dataset
x_train, x_test, y_train, y_test = train_test_split(data, iris, test_size=0.33, random_state=42)

# initilaze the model and train the model afterthat predict the test dataset
model = KNeighborsClassifier(n_neighbors=3, weights='distance')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred_prob = model.predict_proba(x_test)


# confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 0], pos_label="Iris-setosa")
print(f"""
FPR:
{fpr}

TPR:
{tpr}

Thresholds:
{thresholds}
""")
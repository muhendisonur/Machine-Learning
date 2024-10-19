import re 
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('Restaurant_Reviews.csv', on_bad_lines='skip')

nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

preproccesed = []

for i in range(df.shape[0]):
    sentence = re.sub('[^a-zA-Z]', ' ', df.iloc[i,0]).strip().split()
    sentence = [ps.stem(word) for word in sentence if not word in set(stopwords.words("english"))]
    preproccesed.append(' '.join(sentence))
    
cv = CountVectorizer(max_features=2000)
words_count = cv.fit_transform(preproccesed).A
liked_situation = df.iloc[:, 1].fillna(0).values # filled very little part of the dataset with 0


x_train, x_test, y_train, y_test = train_test_split(words_count, liked_situation, test_size=0.33, random_state=61)



model = RandomForestClassifier()
model.fit(x_train, y_train)
pred = model.predict(x_test)

cm = confusion_matrix(y_test, pred) 
accuracy = (cm[0,0] + cm[1,1]) / len(pred) * 100
print(f"""
Confusion Matrix:
{cm}

Accuracy:
%{accuracy}
""")



import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv('TrainingDataBinary.csv')
X = dataset.iloc[:, :-13]
Y = dataset.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
model = RandomForestClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
accuracy = model.score(X_test, Y_test)
print("Accuracy:", accuracy)




import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
training_data = pd.read_csv("TrainingDataMulti.csv")
X = training_data.iloc[:, :-1]
y = training_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
train_accuracy = accuracy_score(y_pred, y_test)
print("Training Accuracy:", train_accuracy)
training_data = pd.read_csv("TestingResultsMulti.csv")
print(y_pred)
new_data = pd.read_csv("TestingDataMulti.csv",header=None)
tdata=new_data.iloc[:, :]
new_data_pred = rf_classifier.predict(tdata)
print(new_data_pred)
results_df = pd.DataFrame(new_data_pred)
results_df.to_csv("NewDataPredictions.csv", index=False)
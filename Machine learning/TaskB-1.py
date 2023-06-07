
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
training_data = pd.read_csv("TrainingDataMulti.csv")
X_train = training_data.iloc[:, :-1]  
y_train = training_data.iloc[:, -1] 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
y_pred_val = rf_classifier.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", val_accuracy)
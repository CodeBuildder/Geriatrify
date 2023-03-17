import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv(r"C:\Users\Madhumitha\Desktop\Majorrrrrr\Health-App\data\heart.csv")

categorical_val = []
continous_val = []
for column in df.columns:
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)

categorical_val.remove('target')
dataset = pd.get_dummies(df, columns = categorical_val)

cols = ['cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang']       
X = df[cols]
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))

model = LogisticRegression()
model.fit(X_train, y_train)
lr_pred = model.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, lr_pred)))

clf_report = classification_report(y_test, lr_pred)
print('Classification report')
print("---------------------")
print(clf_report)
print("_____________________")

joblib.dump(model,r"C:\Users\Madhumitha\Desktop\Majorrrrrr\Health-App\Heart_API\heart_model.pkl")


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import joblib

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

data = pd.read_csv(r"C:\Users\Madhumitha\Desktop\Major Project\dataset\Stroke_Data\stroke-data.csv")
data.drop('id', axis =1, inplace = True)
mean_value=data['bmi'].mean()
data['bmi'].fillna(value=mean_value, inplace=True)

encoder = LabelEncoder()

column = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for i in column:
    data[i] = encoder.fit_transform(data[i])

from sklearn.utils import resample
majority = data[data['stroke'] == 0]
minority = data[data['stroke'] == 1]
upsampled = resample(minority, replace=True, n_samples=len(majority))

strokedata = pd.concat([majority,upsampled])
strokedata = strokedata.sample(frac=1).reset_index(drop=True)

X = strokedata.drop(['stroke'], axis = 1)
y = strokedata['stroke']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

clf_report = classification_report(y_test, y_pred)
print('Classification report')
print("---------------------")
print(clf_report)
print("_____________________")

joblib.dump(model,r"C:\Users\Madhumitha\Desktop\Major Project\Model\Stroke_API\stroke_model.pkl")



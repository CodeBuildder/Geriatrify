import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle

df = pd.read_csv(r"C:\Users\Madhumitha\Desktop\MajorProject\dataset\Heart_data\heart_failure.csv")

X=df.drop(['DEATH_EVENT'],axis=1)
y=df['DEATH_EVENT']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=2)

model=RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
confusion_svc=confusion_matrix(y_test,model.predict(X_test))
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

clf_report = classification_report(y_test, y_pred)
print('Classification report')
print("---------------------")
print(clf_report)
print("_____________________")

filename = r'C:\Users\Madhumitha\Desktop\MajorProject\Model\Heart_API\heart_model.pkl'
pickle.dump(model, open(filename, 'wb'))

# pickle.dump(model,r"C:\Users\Madhumitha\Desktop\Major Project\Model\Heart_API\heart_model.pkl")
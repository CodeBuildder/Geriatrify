import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv(r"C:\Users\Madhumitha\Desktop\Major Project\dataset\Lung_data\lung_cancer.csv")

df.drop_duplicates(inplace=True)


encoder = LabelEncoder()
df['LUNG_CANCER']=encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER']=encoder.fit_transform(df['GENDER'])

df.drop(df[df['AGE'] <= 55].index, inplace = True)

con_col = ['AGE']
cat_col=[]
for i in df.columns:
    if i!='AGE':
        cat_col.append(i)

X=df.drop(['LUNG_CANCER'],axis=1)
y=df['LUNG_CANCER']

for i in X.columns[2:]:
    temp=[]
    for j in X[i]:
        temp.append(j-1)
    X[i]=temp


X_over,y_over=RandomOverSampler().fit_resample(X,y)


X_train,X_test,y_train,y_test = train_test_split(X_over,y_over,random_state=42,stratify=y_over)


scaler=StandardScaler()
X_train['AGE']=scaler.fit_transform(X_train[['AGE']])
X_test['AGE']=scaler.transform(X_test[['AGE']])


param_grid={'C':[0.001,0.01,0.1,1,10,100], 'gamma':[0.001,0.01,0.1,1,10,100]}
model=RandomizedSearchCV(SVC(),param_grid,cv=5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
confusion_svc=confusion_matrix(y_test,model.predict(X_test))
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

clf_report = classification_report(y_test, y_pred)
print('Classification report')
print("---------------------")
print(clf_report)
print("_____________________")

joblib.dump(model,r"C:\Users\Madhumitha\Desktop\Major Project\Model\Lung_API\lung_model.pkl")

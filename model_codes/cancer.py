import pandas as pd
import numpy as np
import joblib

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv(r"E:\External Projects\MajorProject\data\cancer.csv")
df.rename(columns={'GENDER': 'GENDER', 'AGE': 'AGE', 'SMOKING': 'SMOKING', 'YELLOW_FINGERS': 'YELLOW_FINGERS', 'ANXIETY': 'ANXIETY',
                   'PEER_PRESSURE': 'PEER_PRESSURE', 'CHRONIC DISEASE': 'CHRONIC DISEASE', 'FATIGUE ': 'FATIGUE', 'ALLERGY ': 'ALLERGY', 'WHEEZING': 'WHEEZING',
                   'ALCOHOL CONSUMING': 'ALCOHOL CONSUMING', 'COUGHING': 'COUGHING', 'SHORTNESS OF BREATH': 'SHORTNESS OF BREATH',
                   'SWALLOWING DIFFICULTY': 'SWALLOWING DIFFICULTY', 'CHEST PAIN': 'CHEST PAIN', 'LUNG_CANCER': 'LUNG_CANCER'}, inplace=True)

df = pd.get_dummies(data=df, columns=['LUNG_CANCER'], drop_first=True)
df1 = df.groupby('AGE').agg({'GENDER': 'count', 'SMOKING': 'sum', 'YELLOW_FINGERS': 'sum', 'ANXIETY': 'sum',
                             'PEER_PRESSURE': 'sum', 'CHRONIC DISEASE': 'sum', 'FATIGUE': 'sum', 'ALLERGY': 'sum', 'WHEEZING': 'sum',
                             'ALCOHOL CONSUMING': 'sum', 'COUGHING': 'sum', 'SHORTNESS OF BREATH': 'sum',
                             'SWALLOWING DIFFICULTY': 'sum', 'CHEST PAIN': 'sum', 'LUNG_CANCER_YES': 'sum'})
df2 = df.groupby('GENDER').agg({'AGE': 'count', 'SMOKING': 'sum', 'YELLOW_FINGERS': 'sum', 'ANXIETY': 'sum',
                                'PEER_PRESSURE': 'sum', 'CHRONIC DISEASE': 'sum', 'FATIGUE': 'sum', 'ALLERGY': 'sum', 'WHEEZING': 'sum',
                                'ALCOHOL CONSUMING': 'sum', 'COUGHING': 'sum', 'SHORTNESS OF BREATH': 'sum',
                                'SWALLOWING DIFFICULTY': 'sum', 'CHEST PAIN': 'sum', 'LUNG_CANCER_YES': 'sum'})
df = pd.get_dummies(data=df, columns=['GENDER'], drop_first=True)
df.drop_duplicates(inplace=True)
df3 = df.groupby('SMOKING').agg({'GENDER_M': 'sum', 'AGE': 'count', 'SMOKING': 'sum', 'YELLOW_FINGERS': 'sum', 'ANXIETY': 'sum',
                                 'PEER_PRESSURE': 'sum', 'CHRONIC DISEASE': 'sum', 'FATIGUE': 'sum', 'ALLERGY': 'sum', 'WHEEZING': 'sum',
                                 'ALCOHOL CONSUMING': 'sum', 'COUGHING': 'sum', 'SHORTNESS OF BREATH': 'sum',
                                 'SWALLOWING DIFFICULTY': 'sum', 'CHEST PAIN': 'sum', 'LUNG_CANCER_YES': 'sum'})
df4 = df.groupby('YELLOW_FINGERS').agg({'GENDER_M': 'sum', 'AGE': 'count', 'SMOKING': 'sum', 'YELLOW_FINGERS': 'sum', 'ANXIETY': 'sum',
                                        'PEER_PRESSURE': 'sum', 'CHRONIC DISEASE': 'sum', 'FATIGUE': 'sum', 'ALLERGY': 'sum', 'WHEEZING': 'sum',
                                        'ALCOHOL CONSUMING': 'sum', 'COUGHING': 'sum', 'SHORTNESS OF BREATH': 'sum',
                                        'SWALLOWING DIFFICULTY': 'sum', 'CHEST PAIN': 'sum', 'LUNG_CANCER_YES': 'sum'})
df5 = df.groupby('ANXIETY').agg({'GENDER_M': 'sum', 'AGE': 'count', 'SMOKING': 'sum', 'YELLOW_FINGERS': 'sum', 'ANXIETY': 'sum',
                                 'PEER_PRESSURE': 'sum', 'CHRONIC DISEASE': 'sum', 'FATIGUE': 'sum', 'ALLERGY': 'sum', 'WHEEZING': 'sum',
                                 'ALCOHOL CONSUMING': 'sum', 'COUGHING': 'sum', 'SHORTNESS OF BREATH': 'sum',
                                 'SWALLOWING DIFFICULTY': 'sum', 'CHEST PAIN': 'sum', 'LUNG_CANCER_YES': 'sum'})
scaler = StandardScaler()
scaler.fit(df.drop('LUNG_CANCER_YES', axis=1))
scaled_features = scaler.transform(df.drop('LUNG_CANCER_YES', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                                                 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
                                                 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                                                 'SWALLOWING DIFFICULTY', 'CHEST PAIN',  'GENDER_M'])
X = df_feat

y = df['LUNG_CANCER_YES']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

joblib.dump(
    knn, r"E:\External Projects\MajorProject\Cancer_API\cancer_model.pkl")


#

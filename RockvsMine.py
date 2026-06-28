#importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data Collection and Data Processing
#loading the dataset to pandas DataFrame
sonar_data=pd.read_csv("Sonar_ML/Copy of sonar data.csv",header=None,encoding="unicode_escape")
# print(sonar_data.head())
# print(sonar_data.shape)
# print(sonar_data.describe())
# print(sonar_data[60].value_counts())#gives no of counts for mine and rock
# print(sonar_data.groupby(60).mean())
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]
print(X)
print(Y)

#Traing and Testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
print(X.shape,X_train.shape,X_test.shape)
print(X_train)
#Y_train is lables of traing data(X_train)
print(Y_train)

#Model traing-->Logistic Regression
model=LogisticRegression(max_iter=1000)
#traing the model with traing data
print(model.fit(X_train,Y_train))
#accuracy on traing data
X_train_prediction=model.predict(X_train)
traing_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy on traing data: ",traing_data_accuracy)

#accuracy on testing data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy on testing data: ",test_data_accuracy)

#Making a predictive system
input_data=(0.0238,0.0318,0.0422,0.0399,0.0788,0.0766,0.0881,0.1143,0.1594,0.2048,0.2652,0.31,0.2381,0.1918,0.143,0.1735,0.1781,0.2852,0.5036,0.6166,0.7616,0.8125,0.7793,0.8788,0.8813	,0.947,1,0.9739,0.8446,0.6151,0.4302,0.3165,0.2869,0.2017,0.1206,0.0271,0.058,0.1262,0.1072,0.1082,0.036,0.1197,0.2061,0.2054,0.1878,0.2047,0.1716,0.1069,0.0477,0.017,0.0186,0.0096,0.0071,0.0084,	0.0038,	0.0026,	0.0028,	0.0013,	0.0035,	0.006
)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=='R'):
    print("The object is Rock")
else:
    print("The object is a mine")




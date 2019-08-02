#Simple Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading the dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

'''
#Feature Scaling
#Xstd=(X-mean(x))/std(x)
#Xnorm=(X-Xmin)/(xmax-xmin)
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train)
'''
#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred=regressor.predict(X_test)

#Visualising the training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Exp Vs Sal (Training Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#Visualing the test set
plt.scatter(X_test,y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Exp Vs Sal (Test Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()


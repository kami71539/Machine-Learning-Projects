#Polynomial Regression
#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,-1].values


#Splitting the dataset
'''
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
'''
#Feature Scaling
'''
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.fit_transform(X_test)
sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train)
'''
#Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

# Fittin polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(4)
X_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)
y2_pred=lin_reg2.predict(X_poly)

#Visualing the linear regression plot
plt.scatter(X,y,color='red')
plt.plot(X,y_pred,color='blue')
plt.title("Position vs Salary, (Linear Regression)")
plt.xlabel('Position')
plt.ylabel("Salary")
plt.show()

#Visualing the polynomial regression plot
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title=("Position vs Salary (Polynomial Regression)")
plt.xlabel("Salary")
plt.ylabel("Position")
plt.show()

#Prediction of new result using Linear regression model
prediction_of_linear_regression=lin_reg.predict(np.array(6.5).reshape(-1,1))

#Prediction of new result using Polynomial regression
Prediciton_of_polynomial_regressiosn=lin_reg2.predict(poly_reg.fit_transform(np.array(6.5).reshape(-1,1)))

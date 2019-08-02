#Random Forest

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Training the model
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X,y)

#Visualing the result
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title("Random Forest Regression")
plt.xlabel("Position Level")
plt.ylabel("Expected Salary")
plt.show()

#Predicting the result
y_pred=regressor.predict(np.array(6.5).reshape(-1,1))
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 18:06:04 2022

@author: 20050
"""

import pandas as pd
df=pd.read_csv("C:\\Data Science\\Assignments\\delivery_time.csv")
df

###spilt the variable

X=df[["Sorting Time"]]
X
Y=df["Delivery Time"]
Y

## EDA (Scatter plot)

import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0],Y,color='blue')
plt.ylabel("Delivery Time")
plt.xlabel("Sorting Time")
plt.show()

##Model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y2=LR.predict(X)
Y2

##Calculate RMSE,R square

from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
mse=mean_squared_error(Y,Y2)
RMSE=np.sqrt(mse)
print("Root mean square value:",RMSE)
print("R square value:",r2_score(Y,Y2))























# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:29:38 2019

@author: pais
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')


X= dataset.iloc[:, 1:2].values
Y=dataset.iloc[:, 2:3].values


#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=0)
#
# Fitting linear model
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X,Y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
X_poly= poly_reg.fit_transform(X)

lin_reg2= LinearRegression()
lin_reg2.fit(X_poly,Y)

#visualization
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or bluff(linearregg)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

#poly
X_grid=np.arange(min(X),max(X),0.1)
X_grid= X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or bluff(polyrregg)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
Z=[[6.5],[8]]
#Z=Z.array.reshape(-1,1)
lin_reg.predict(Z)

lin_reg2.predict(poly_reg.fit_transform(Z))



#Authors: Thiago Emanuel and Julio C. S. Da Silva
# To get the used data:
# wget https://raw.githubusercontent.com/jcsdasilva/Pt-T/main/Pt-T-TS-Data-ML.csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('Pt-T-TS-Data-ML.csv')
print(list(data))

x = data[['Pt-L', 'Pt-T', 'Pt-NH3']]
N = len(x)
print(N)
ones = np.ones(N)
xp = np.c_[ones,x]
x


x = data[['Pt-L', 'Pt-T', 'Pt-NH3']]
N = len(x)
print(N)
ones = np.ones(N)
xp = np.c_[ones,x]
y = data['Gact']
y = y.values.reshape(1,-1)

np.random.seed(0) # dont change the random generator and to not affetc the learning rate
w = 2*np.random.rand(4) - 1 # Generates ramdom numbers 


epochs = 100000000 #Changes epochs to reach convergence
learning_rate = 0.0013
for epoch in range(epochs):
  y_predicted = w @ xp.T
  error = y - y_predicted
  L2 = 0.5*np.mean(error**2)
  gradient = -(1/10)*error @ xp
  w = w - learning_rate*gradient
  if epoch%(epochs/10) == 0:
    print(epoch, L2)
print(w)
print(y_predicted)


from sklearn.linear_model import LinearRegression
x = data[['Pt-L', 'Pt-T', 'Pt-NH3']]
y = data['Gact']
reg = LinearRegression().fit(x,y)
print(reg.coef_)
print(reg.intercept_)
y_predicted = reg.predict(x)
error = y - y_predicted
L2 = 0.5*np.mean(error**2)
print(L2)
print(y_predicted)

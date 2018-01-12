import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#data
points = np.genfromtxt('challenge_dataset.txt', delimiter = ',')
x_values = points[:,0:1]
y_values = points[:,1:2]

#train
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()


import numpy as np
from sklearn.linear_model import LinearRegression

math = np.array([80, 82, 65, 45, 72, 66, 68, 90, 83, 77]).reshape(-1, 1)
physics = np.array([90, 95, 71, 42, 88, 72, 76, 94, 83, 82]).reshape(-1, 1)

model = LinearRegression()
model.fit(math, physics)
model.predict([[77]])
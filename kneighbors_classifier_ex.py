import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = np.loadtxt("datasets/ex_class.csv", delimiter=",", skiprows=1)

x = data[:,1:]
y = data[:,0]

x_train, x_test, y_train, y_test = train_test_split(x, y)
model = KNeighborsClassifier()

model.fit(x_train, y_train)
print(model.predict(x_test))
print(y_test)
model.score(x_test, y_test)
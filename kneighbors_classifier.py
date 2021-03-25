import numpy as np
from sklearn.neighbors import KNeighborsClassifier

train = np.loadtxt("datasets/train01.csv", delimiter=",", skiprows=1)
test = np.loadtxt("datasets/test01.csv", delimiter=",", skiprows=1)

y_train = train[:,0]
x_train = train[:,1:]

x_test = test[:,1:]
y_test = test[:,0]

clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

pre = clf.predict(x_test)
print(pre)
print(y_test)
print(clf.score(x_test, y_test))
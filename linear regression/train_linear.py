import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear import LinearRegression 

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fig = plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
# plt.show()

clf = LinearRegression(lr = 0.05)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test - predictions)**2)

mse = mse(y_test, predictions)
print("mse: " + str(mse))

y_mean = np.mean(y_test)
def r_sq(y_test, predictions, y_mean):
    return 1 - (np.sum((y_test - predictions)**2))/(np.sum((y_test - y_mean)**2))

r_sq = r_sq(y_test, predictions, y_mean)
print("r_squared: " + str(r_sq))

y_pred_line = clf.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5))
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()

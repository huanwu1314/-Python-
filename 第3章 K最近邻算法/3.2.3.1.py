from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import numpy as np 


X,y = make_regression(n_features=1,n_informative=1,noise=50,random_state=8)
plt.scatter(X,y,c='orange',edgecolor='k')
reg= KNeighborsRegressor()
reg.fit(X,y)
z = np.linspace(-3,3,200).reshape(-1,1)
plt.plot(z,reg.predict(z),c='k',linewidth=3)
plt.title('KNN Regressor')
plt.show()
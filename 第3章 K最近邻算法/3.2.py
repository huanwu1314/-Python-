import numpy as np 
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = make_blobs (n_samples=200,centers=2,random_state=8)
X,y=data
clf = KNeighborsClassifier()
clf.fit(X,y)

#下面代码用于画图
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z,cmap =plt.cm.Pastel1)
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolor='k')
plt.xlim(xx.min(),xx.max())
plt.xlim(yy.min(),yy.max())
plt.title("Classifier:KNN")

#把新的数据点用五星表示出来 
plt.scatter(6.75,4.82,marker='*',c='red',s=200)
plt.show()

print('\n\n\n')
print('代码运行结果：')
print('==============================')
print('新数据点分类是：',clf.predict([[6.75,4.82]]))
print('==============================')
print('\n\n\n')

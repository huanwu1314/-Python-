from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np 
# 生成样本数为50，分类数为5的数据集
data2 = make_blobs(n_samples=500, centers=5,random_state=8)
X2,y2=data2

clf = KNeighborsClassifier()
clf.fit(X2,y2)

#下面代码用于画图
x_min, x_max = X2[:,0].min()-1, X2[:,0].max()+1
y_min, y_max = X2[:,1].min()-1, X2[:,1].max()+1
xx, yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z,cmap =plt.cm.Pastel1)
plt.scatter(X2[:,0],X2[:,1],c=y2,cmap=plt.cm.spring,edgecolor='k')
plt.xlim(xx.min(),xx.max())
plt.xlim(yy.min(),yy.max())
plt.title("Classifier:KNN")
plt.show()
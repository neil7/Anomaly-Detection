# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:27:32 2018

@author: Neil sharma
"""


# %load ../../../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split
#import collections
from sklearn.cross_validation import train_test_split
#from keras.utils import to_categorical
#from keras import utils as np_utils


pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)
 
#%config InlineBackend.figure_formats = {'pdf',}
#%matplotlib qt

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')


def plotData(X, y):
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    
    plt.scatter(X[pos,0], X[pos,1], s=60, c='k', marker='+', linewidths=1)
    plt.scatter(X[neg,0], X[neg,1], s=60, c='y', marker='o', linewidths=1)
    
    
def plot_svc(svc, X, y, h=0.02, pad=0.25):
#    X[1] = X[1].reshape(12)
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xx, yy)
    Xpred = np.array([xx.ravel(), yy.ravel()] + [np.repeat(0, xx.ravel().size) for _ in range(10)]).T
#    print(xx.shape)
    Z = svc.predict(Xpred).reshape(xx.shape)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plotData(X, y)
    #plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)

#    rs = np.random.RandomState(1234)    
    #Fake data
#    n_samples = 200
#    a = n_samples//2
#    # X is the input features by row.
#    X = Xpred
#    X[:a] = rs.multivariate_normal( np.ones(3), np.eye(3), size=a)
#    X[a:] = rs.multivariate_normal(-np.ones(3), np.eye(3), size=a)
#    # Y is the class labels for each row of X.
#    Y = np.zeros(n_samples); Y[a:] = 1
#    Z = lambda xx,yy: (-svc.intercept_[0]-svc.coef_[0][0]*xx-svc.coef_[0][1]) / svc.coef_[0][2]
#    fig = plt.figure()
#    ax  = fig.add_subplot(111, projection='3d')
#    ax.plot_surface(xx, yy, Z(xx,yy))
#    ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
#    ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
#    plt.show()
    
    
def gaussianKernel(x1, x2, sigma=2):
    norm = (x1-x2).T.dot(x1-x2)
    return(np.exp(-norm/(2*sigma**2)))

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

gaussianKernel(x1, x2, sigma)

#SVMs
#import h5py 
#data1 = h5py.File(r'D:\ML\Shrey Anomaly detection\Dataset\smtp.mat')

data1 = loadmat(r'D:\ML\Shrey Anomaly detection\Dataset\arrhythmia.mat')
data1.keys()

y1 = data1['y']
X1 = data1['X']

import numpy as np
#X1 = np.array(X1)
#X1 = np.swapaxes(X1, 0, 1)
##X = X.reshape(1, 1)
#y1 = np.array(y1)
#y1 = np.swapaxes(y1, 0, 1)

print('X1:', X1.shape)
print('y1:', y1.shape)

#plotData(X1,y1)

X_train, X_test, y_train, y_test = train_test_split( X1, y1, test_size=0.3, random_state=1 )

clf = SVC(C=150, kernel='rbf')
#clf.set_params(C=150)
clf.fit(X_train, y_train.ravel())
result = clf.predict(X_test)
print (clf.score(X_test,y_test))


#plot_svc(clf, X_train, y_train)


#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, result)

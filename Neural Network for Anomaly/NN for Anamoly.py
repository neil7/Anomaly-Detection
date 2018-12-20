# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:32:24 2018

@author: Neil sharma
"""
# Importing the dataset

#import hdf5storage
#data1 = hdf5storage.loadmat(r'D:\ML\Shrey Anomaly detection\Dataset\smtp.mat')

#import h5py 
#data1 = h5py.File(r'D:\ML\Shrey Anomaly detection\Dataset\smtp.mat')

from scipy.io import loadmat
data1 = loadmat(r'D:\ML\Shrey Anomaly detection\Dataset\breastw.mat')
data1.keys()

y = data1['y']
X = data1['X']


#import numpy as np
#X = np.array(X)
#X = np.swapaxes(X, 0, 1)
##X = X.reshape(1, 1)
#y = np.array(y)
#y = np.swapaxes(y, 0, 1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
import numpy as np
np.save('X_test_Breast.npy', X_test)
np.save('y_test_Breast.npy', y_test)

## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!
input_dim = X_train.shape[1]
n_neurons = int(input_dim / 2)
#n_neurons = 6
# Importing the Keras libraries and packages
#import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = n_neurons, kernel_initializer = 'uniform', activation = 'sigmoid',
                     input_dim = input_dim))

# Adding the second hidden layer
classifier.add(Dense(units = n_neurons, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
batch_size = 10
epochs = 100
validation_data = (X_test, y_test)
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, validation_data = validation_data, batch_size = batch_size, epochs = epochs)

# serialize model to JSON
classifier.save_weights("model.h5")
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
print("Saved model to disk")

# Part 3 - Making predictions and evaluating the model
result = classifier.predict(X_test)
result = result > 0.5
print (classifier.evaluate(X_test,y_test))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, result)
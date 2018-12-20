# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:10:41 2018

@author: Neil sharma
"""
import numpy as np

# Importing the Keras libraries and packages
from keras.models import model_from_json


# load json and create model
json_file = open(r'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights(r"model.h5")
print("Loaded model from disk")

#Load testing dataset from the numpy array
X_test = np.load('X_test_Breast.npy')
y_test = np.load('y_test_Breast.npy')

# Part 3 - Making predictions and evaluating the model
result = classifier.predict(X_test)
result = result > 0.5
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print (classifier.evaluate(X_test,y_test))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, result)
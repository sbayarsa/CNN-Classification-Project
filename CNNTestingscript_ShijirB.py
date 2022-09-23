# Shijir Bayarsaikhan
# sbayarsa@vt.edu
# ECE 5984 AI Project
# Spring 2020 Semester
# CNN Testing Script
#
# Application of the Convolutional Neural Network Architecture for Image Classification using Custom Created Images Composed of Digitally Drawn Digits
# Some Code Contributions from:
#  -Assignment 3 MNIST Tensorflow testing script and 
#  -Assignment 2 Questions 2-3 script
#  -Yal, O.
#  -Sharma, A. 
#  -Rajaraman, S.
#  -Harrison, H.
#  -Gouillart, E.
#  -Brownlee, J.
#  -Bhobe, M.
#
# Full list of code contributions sources provided in Appendix section of the Final Report
#
#---------------------------------------------------------------------------------------------------------------------------------------
#
# Libraries and Packages required to run the testing script:
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.np_utils import to_categorical
#
# Defining the Path and Importing the data and creating lists for images and classes
path ='TestingData'
images = [] #list all images
classes = [] #folders/classes for each i.e 1 = images of 1 
TrainingFolderList = os.listdir(path)
NumberOfClasses = len(TrainingFolderList)
#
# Reading the images and appending to list
for x in range (0, NumberOfClasses):
    imagefiles = os.listdir(path+"/"+str(x))
    for y in imagefiles:
        img = cv2.imread(path+"/"+str(x)+"/"+y)
        images.append(img)
        classes.append(x)
    print (x)
#
# Turning into Arrays
images = np.array(images)
classes = np.array(classes)
#
# Defining variables for images and classes from test dataset
x_test_data = images
y_test_classes = classes
#
# Selecting 12 random images from the test dataset to evaluate on using the trained model
rand_12 = np.random.randint(0, x_test_data.shape[0],12)
sample_digits = x_test_data[rand_12]
sample_labels = y_test_classes[rand_12]	 
#
# Image PreProcessing for the 12 randomly selected images from test dataset
def Preproccesimages(img):
    img = cv2.resize(img,(28,28))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img/255
    return img
sample_digits = np.array(list(map(Preproccesimages,sample_digits)))
x_test_data = np.array(list(map(Preproccesimages,x_test_data)))
#
# Reshaping the 12 randomly selected images from test dataset 
sample_digits = sample_digits.reshape(sample_digits.shape[0],sample_digits.shape[1], sample_digits.shape[2],1)
x_test_data=x_test_data.reshape(x_test_data.shape[0],x_test_data.shape[1], x_test_data.shape[2],1)
#
# Convert class integers to binary class matrices
y_test_classes = to_categorical(y_test_classes,NumberOfClasses)
sample_labels = to_categorical(sample_labels,NumberOfClasses)
#
#Loading the trained CNN Model
model = tf.keras.models.load_model("trainedmodel_ShijirB.h5")
#
# Evaluating the test dataset 
test_results = model.evaluate(x_test_data, y_test_classes)
#
#Printing the results and performance
print('Test Accuracy =', test_results[1]*100)
#
# Making Predictions on the 12 randomly selected images from test dataset
predictions = model.predict(sample_digits)
tf.print(predictions.shape)
predictions = tf.argmax(predictions, axis=1)
print(predictions)
#
# Visualizing the predictions for the 12 randomly selected images from test dataset including predicted class labels
fig=plt.figure(figsize=(12,4))
for i in range(0,12):
    image = sample_digits[i]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28,28))
    ax = fig.add_subplot(2, 6, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(pixels, cmap='gray_r') 
    ax.text(0.7, 0.1, '{}'.format(predictions[i]),
	        size=10, color='blue',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
plt.show()
print("Done")
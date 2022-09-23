# Shijir Bayarsaikhan
# sbayarsa@vt.edu
# ECE 5984 AI Project
# Spring 2020 Semester
# CNN Training Script
#
# Application of the Convolutional Neural Network Architecture for Image Classification using Custom Created Images Composed of Digitally Drawn Digits
# Some Code Contributions from:
#  -Assignment 3 MNIST Tensorflow training script and 
#  -Assignment 2 Questions 2-3 script
#  -Yal, O.
#  -Sharma, A. 
#  -Rajaraman, S.
#  -Mack, D.
#  -Harrison, H.
#  -Gouillart, E.
#  -Gandhi, R.
#  -Brownlee, J.
#  -Bhobe, M.
#  -Alberola, A.
#
# Full list of code contributions sources provided in Appendix section of the Final Report
#
#---------------------------------------------------------------------------------------------------------------------------------------
#
# Libraries and Packages required to run the training script:
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from timeit import default_timer as timer
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
#
# CNN Model Parameters:
imageDim= (28,28,3)
NumberOfFilters= 64
Filter1Size = (5,5)
Filter2Size = (3,3)
PoolSize = (2,2)
NumberOfNodes = 100
BatchSizeNumber = 10
StepsPerEpoch= 400
NumberOfEpoch = 20
#
# Training and Validation Split Ratio Parameter
ValidationRatio = 0.20
#
# Defining the Path and Importing the data and creating lists for images and classes 
path ='TrainingData'
images = [] #list all images
classes = [] #folders/classes for each i.e 1 = images of 1 
#
# Showing Number of Classes, 0-9 = 10 Total Classes
TrainingFolderList = os.listdir(path)
print("Number of Classes", len(TrainingFolderList))
NumberOfClasses = len(TrainingFolderList)
#
for x in range (0, NumberOfClasses):
    imagefiles = os.listdir(path+"/"+str(x))
    for y in imagefiles:
        img = cv2.imread(path+"/"+str(x)+"/"+y)
        images.append(img)
        classes.append(x)
    print (x)
#
# Checking Number of data
print(len(classes))	
print(len(images))
#
# Turning into Arrays
images = np.array(images)
classes = np.array(classes)
#
print(images.shape)
print(classes.shape)
#
# Spliting Data to Training and Validation Sets
x_train_data, x_train_validation, y_train_classes, y_train_validation = train_test_split(images,classes,test_size=ValidationRatio,random_state=None)
print(x_train_data.shape)
print(y_train_classes.shape)
print(x_train_validation.shape)
print(y_train_validation.shape)
#
# Showing the selected Images Distribution with plot bar
NumberOfSelectedImages= []
for x in range(0,NumberOfClasses):
    #print(len(np.where(y_train==x)[0]))
    NumberOfSelectedImages.append(len(np.where(y_train_classes==x)[0]))
print(NumberOfSelectedImages)
#
plt.figure(figsize=(10,5))
plt.bar(range(0,NumberOfClasses),NumberOfSelectedImages)
plt.title("Images for each Class")
plt.xlabel("Class Number")
plt.ylabel("Number of Images")
plt.show()
#
tf.random.set_seed(1234)
#
# Showing 10 randomly selected images from the training dataset with proper labels included
rand_10 = np.random.randint(0, x_train_data.shape[0],10)
sample_digits = x_train_data[rand_10]
sample_labels = y_train_classes[rand_10]
num_rows, num_cols = 2, 5
f, ax = plt.subplots(num_rows, num_cols, figsize = (13,5),
                     gridspec_kw = {'wspace':0.05, 'hspace': 0.13},
                     squeeze=True)                   
for r in range(num_rows):
    for c in range(num_cols):
        image_index = r*5+c
        ax[r,c].axis("off")
        ax[r,c].imshow(sample_digits[image_index],cmap='gray')
        ax[r,c].set_title('No. %d'% sample_labels[image_index])
plt.show()
plt.close()
#
# Image PreProcessing
def Preproccesimages(img):
    img = cv2.resize(img,(28,28))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img/255
# #   img = img.reshape(28,28,1)
    return img    
#    
x_train_data = np.array(list(map(Preproccesimages,x_train_data)))
x_train_validation = np.array(list(map(Preproccesimages,x_train_validation)))
#
# Reshaping Images needed for CNN
x_train_data=x_train_data.reshape(x_train_data.shape[0],x_train_data.shape[1], x_train_data.shape[2],1)
x_train_validation=x_train_validation.reshape(x_train_validation.shape[0],x_train_validation.shape[1], x_train_validation.shape[2],1)
print (x_train_data.shape)	
print (x_train_validation.shape)
#
# Image Augmentation to help with Overfitting
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(x_train_data)
#
# Convert class integers to binary class matrices
y_train_classes = to_categorical(y_train_classes,NumberOfClasses)
y_train_validation = to_categorical(y_train_validation, NumberOfClasses)
print (y_train_classes.shape) 
print (y_train_validation.shape)
#
# Building the CNN Model
model = Sequential()
model.add((Conv2D(NumberOfFilters,Filter1Size,input_shape=(imageDim[0],
                     imageDim[1],1),activation='relu')))
model.add((Conv2D(NumberOfFilters, Filter1Size, activation='relu')))
model.add(MaxPooling2D(pool_size=PoolSize))
model.add((Conv2D(NumberOfFilters//2,Filter2Size, activation='relu')))
model.add(MaxPooling2D(pool_size=PoolSize))
model.add(Flatten())
model.add(Dense(NumberOfNodes,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(NumberOfClasses, activation='softmax'))
#
tf.random.set_seed(1234)
model.summary()
model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
#
# Starting the timer and Training the Model
startTime = timer()
history = model.fit_generator(dataGen.flow(x_train_data,y_train_classes,
                                 batch_size=BatchSizeNumber),
                                 steps_per_epoch=StepsPerEpoch,
                                 epochs=NumberOfEpoch,
                                 validation_data=(x_train_validation,y_train_validation),
                                 shuffle=1)
#														 
hist = history.history
x_arr = np.arange(len(hist['loss'])) + 1
endTime = timer()	
print ('It took ', (endTime - startTime)/60, ' minutes to train the model.')
#
# Visualizing the Results
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
ax.plot(x_arr, hist['val_loss'], '--<', label='Validation loss')
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
plt.show()
#
# Saving the Model
model.save("trainedmodel_ShijirB.h5", overwrite=True, include_optimizer=True)
#
print("done")
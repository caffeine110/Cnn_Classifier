#!/usr/bin/env pyth on3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:06:22 2018

@author : Gaurav Gahukar


Aim : Convulational Neural Network to distinguish between two animals

"""

#Phase - 1:  Import
#importing all the required librariees
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#Phase - 2: Building Model
#Initialise the CNN
model_cnn_classifier = Sequential()


#Convolutional step
model_cnn_classifier.add(Convolution2D(32,3,3,  input_shape = (64,64,3), activation= 'relu'))

#Pooling step
model_cnn_classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flatten the step
model_cnn_classifier.add(Flatten())
 
#Full connection
model_cnn_classifier.add(Dense(output_dim = 128, activation= 'relu'))


#Phase - (middle): Output layer 
#model_cnn_classifier.add(Dense(   ))
model_cnn_classifier.add(Dense(output_dim = 1, activation= 'sigmoid'))




#Phase - 3 :  compiling the CNN
model_cnn_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Phase - 4 : Fitting CNN to the image
#Importing 
from keras.preprocessing.image import ImageDataGenerator

# Train Datagen
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range=0.2,
        horizontal_flip=True )
#Test Daragen
test_datagen = ImageDataGenerator(rescale=1./255)

#Training Set
training_set= train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode='binary')

#test Set
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size = (64,64),
        batch_size = 32,
        class_mode='binary')

#Final Step
model_cnn_classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

#!/usr/bin/env pyth on3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:06:22 2018

@author : Gaurav Gahukar
        : Caffeine110

Aim : Convulational Neural Network 

"""


#importing all the required librariees
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialise the CNN
cnn_classifier = Sequential()

#Convolutional step - 1
cnn_classifier.add(Convolution2D(32,3,3,  input_shape = (64,64,3), activation= 'relu'))

#Pooling step - 2
cnn_classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flatten the step -3
cnn_classifier.add(Flatten())
 
#Full connection -4
cnn_classifier.add(Dense(output_dim = 128, activation= 'relu'))

# output layer 
#cnn_classifier.add(Dense(   ))
cnn_classifier.add(Dense(output_dim = 1, activation= 'sigmoid'))


#compiling the CNN
cnn_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#stage 2 Fitting cnn to ithe image
#from keras 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range=0.2,
        horizontal_flip=True )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set= train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode='binary')


test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size = (64,64),
        batch_size = 32,
        class_mode='binary')

cnn_classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
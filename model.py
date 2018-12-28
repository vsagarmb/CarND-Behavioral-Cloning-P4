"""A neural network used for behavioral cloning"""
# for building the neural net
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, ELU, Cropping2D

# for data wrangling
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

import random

images = []
angles = []

def gather_training_data(images, angles, log_path='./data/'):
    '''
    This function reads the csv file at the given path and loads 
    the images and the steering angles into the sample set. 
    '''
    # first gather the default data
    with open(log_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        lines = [line for line in reader]
    
    for line in lines[1:]:
        measurement = float(line[3])
        
        # remove 50% of the driving straight images
        if abs(measurement) <= 0.02 and random.random() < 0.50:
            continue

        # Center Image
        tokens = line[0].split('\\')
        impath = log_path + 'IMG/' + tokens[-1]    
        
        image = cv2.imread(impath)
        
        if image is None:
            print(impath)
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
       
        angles.append(measurement)

        # Left Image
        tokens = line[1].split('\\')
        impath = log_path + 'IMG/' + tokens[-1]    
        
        image = cv2.imread(impath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
       
        angles.append(measurement+0.2)

        # Right Image
        tokens = line[2].split('\\')
        impath = log_path + 'IMG/' + tokens[-1]    
        
        image = cv2.imread(impath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
       
        angles.append(measurement-0.2)

    return images, angles

  
''' 
    ---------------------------------------------------------------------------
    MAIN
    ---------------------------------------------------------------------------
'''


if __name__ == "__main__":
    
    # Extract the data
    images, angles = gather_training_data(images, angles, './data/run1/')
    images, angles = gather_training_data(images, angles, './data/run2/')

    # data marshalling
    X_train = np.array(images)
    y_train = np.array(angles) 
       
    #--------------------------- Neural Network Definition --------------------------------
    #-----This is based on the comm ai model with 2 additional fully connected layers -----

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((75,25), (0,0))))
    model.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), padding="same"))
    model.add(ELU())
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(.5))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    # Compile and Train
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
            
    model.save('model.h5')

    print('Model Saved')

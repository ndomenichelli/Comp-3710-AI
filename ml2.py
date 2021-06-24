# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:44:03 2019

@author: Ndome
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras

# Helper libraries
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

import os

import gc
gc.collect()

#(train_images, train_labels), (test_images, test_labels) = 


#labels for beginning and end segments of the match
my_video_classes= ['start','end1','end2']

dataSet = "FK bSM (Snake) vs. GGs Darkel (CloudHero) - Winner's Finals"

pathTest = "C:\\Users\\Ndome\\Documents\\School\\Comp 3710 AI\\TestVideos - Copy\\"+dataSet+"\\Test"
pathTrain = "C:\\Users\\Ndome\\Documents\\School\\Comp 3710 AI\\TestVideos - Copy\\"+dataSet+"\\Train"
pathTestLabels = "C:\\Users\\Ndome\\Documents\\School\\Comp 3710 AI\\TestVideos - Copy\\"+dataSet+"\\test2.txt"
pathTrainLabels = "C:\\Users\\Ndome\\Documents\\School\\Comp 3710 AI\\TestVideos - Copy\\"+dataSet+"\\train2.txt"
#images = [cv2.imread(file) for file in glob.glob(path+"/*.jpg")]
print(0)


def get_images(pathImage,pathLabel):
    test_images = []
    labels = []
    images = os.listdir(pathImage)
    fileLabel = open(pathLabel, "r")
    
    
    matches = 0
    for image in images:
#        index = 0
        next_Image = cv2.imread(pathImage+ "\\" + image)
        next_Image = np.array(next_Image)
        test_images.append(next_Image)
       # print("nm",next_Image)
        fileLabel = open(pathLabel, "r")
        
#        print(image)
        imageName = image[5:-4]
       # print("imagename", imageName)
        #make empty array for labels same size as images
        count = 0
        for line in fileLabel:
            #print(line)
            
            output = line.split(" ")
            frame = output[0]
            label = output[1]
            
           # print("here", imageName,frame, label)

          
                
            if imageName == frame:
                if(label == "start\n"):
                    labels.append(0)
                elif(label == "end1\n"):
                    labels.append(1)
                elif(label == "end2\n"):
                    labels.append(2)
                matches += 1
                print("Match!")
                count += 1
                
        if count == 0:
            labels.append(4)
            print("no match")
#            print(imageName+ " " + frame)
           # if(imageName == frame):
           #     if(label == "start\n"):
           #         labels.append(0)
           #     elif(label == "end1\n"):
           #         labels.append(1)
           #     elif(label == "end2\n"):
           #         labels.append(2)
           #     break
           # else:
            #    labels.append(4)
        
#        last_pos = fileLabel.tell()
#        line = fileLabel.readline()
#        output = line.split(" ")
#        
#        if(line != ""):
#            frame = output[0]
#            label = output[1]
#            
#            imageName = image[5:-4]
#            print(imageName)
#            if(imageName in line):
#                       
#                if(label == "start\n"):
#                    labels.append(0)
#                elif(label == "end1\n"):
#                    labels.append(1)
#                elif(label == "end2\n"):
#                    labels.append(2)
#                fileLabel.seek(last_pos)
#            else:
#                labels.append(4)
#                
#            last_pos = fileLabel.tell()
                
#            index = index + 1

    labels.append(4)
    print("Matches", matches)
    print("Lengths",len(test_images), len(labels))
    return np.array(test_images), np.array(labels)

def get_labels(length,path):
    labels = []

    file2 = open(path,"r");
    startLine = file2.readline();
    startLine = startLine.split(" ")
    firstLabel = startLine[1]

    for x in range(length):
        
        labels.append(x)

    for line in file2:
        output = line.split(" ")
        frame = output[0]
        label = output[1]
        
        if(label == "start\n"):
            labels[line] = 0
        elif(label == "end1\n"):
            labels[line] = 1
        elif(label == "end2\n"):
            labels[line] = 2

    file2.close() 
    return np.array(labels)
        
#print(os.listdir(path))

imagesTestArray,labelsTestArray = get_images(pathTest,pathTestLabels)
imagesTrainArray,labelsTrainArray = get_images(pathTrain,pathTrainLabels)

print(imagesTestArray.shape)
#print(imagesTrainArray.shape)

#labelsTestArray = get_labels(imagesTestArray.shape[0],pathTestLabels)
print(labelsTestArray.shape)


##
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(360, 640,3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
    
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(imagesTrainArray, labelsTrainArray, epochs=10)

test_loss, test_acc = model.evaluate(imagesTestArray, labelsTestArray, verbose=2)
#
print('\nTest accuracy:', test_acc)

predictions = model.predict(labelsTestArray)

predictions[0]

np.argmax(predictions[0])

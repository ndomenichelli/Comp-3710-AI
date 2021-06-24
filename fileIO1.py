# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:25:44 2019

@author: Ndome
"""
#gets rid of middle chunk of files

from time import sleep

file1 = open("C:\\Users\\Ndome\\Documents\\School\\Comp 3710 AI\\TestVideos - Copy\\FK bSM (Snake) vs. GGs Darkel (CloudHero) - Winner's Finals_gt.txt","r")

#for x in range(10):
#    strToWrite = "test" + str(x) + "\n"
#    file1.write(strToWrite)

file2 = open("C:\\Users\\Ndome\\Documents\\School\\Comp 3710 AI\\TestVideos - Copy\\FK bSM (Snake) vs. GGs Darkel (CloudHero) - Winner's Finals\\train2.txt","w");

for line in file1:
    output = line.split(",")
    strToWrite = output[0] + " " + output[6]
    file2.write(strToWrite)
#    sleep(0.05)

file1.close()
file2.close()
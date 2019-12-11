# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:20:38 2019

@author: Ibram Medhat, Basem Rizk
"""

import random
import numpy as np
from PIL import Image

#Calculates the means of the different segments in the image based on the theresholds
#and pixels in each segement
def recalculate_means(org_image_array, object_detection_img, n) :
    count_each_segment = np.zeros(n+1)
    means_array = np.zeros(n+1)
    for x in range(org_image_array.shape[0]) :
        for y in range(org_image_array.shape[1]) :
            means_array[int(object_detection_img[x][y])] += org_image_array[x][y]
            count_each_segment[int(object_detection_img[x][y])] += 1
    for i in range(n+1) :
        means_array[i] = means_array[i] / count_each_segment[i]
    return means_array

#calculate the theresholds based on the different calculated means
def recalculate_theresholds(means_array) :
    thereshold_array = np.zeros(means_array.size-1)
    for i in range(thereshold_array.size) :
        thereshold_array[i] = (means_array[i] + means_array[i+1]) / 2
    return thereshold_array


def generate_gray_segmented_image(object_detection_img, thereshold_array) :
    gray_image_array = object_detection_img
    for x in range(object_detection_img.shape[0]) :
        for y in range(object_detection_img.shape[1]) :
            gray_image_array[x][y] = object_detection_img[x][y] * (255 / thereshold_array.size+1)
    return gray_image_array

#The function should take as inputs the image to be
#segmented and n. The function should return the computed n thresholds, a binary image for each segment
#and one segmented gray-scale image with each segment assigned a different gray-level
def detect_objects(org_image, n) :
    thereshold_array = np.zeros(n)
    means_array = np.zeros(n+1)
    #Using Random initialization for n theresholds
    for i in range(n) :
        thereshold_array[i] = random.randint(0,255)
    thereshold_array = np.sort(thereshold_array,axis=None)
    print(thereshold_array)
    org_image_array = np.array(org_image)
    object_detection_img = np.zeros(org_image_array.shape[0]*org_image_array.shape[1]).reshape(org_image_array.shape[0], org_image_array.shape[1])
    
    means_changed = True
    
    while(means_changed) :
    
        #grouping the orginal image into different groups according to the theresholds values
        for i in range(n) :
            for x in range(org_image_array.shape[0]) :
                for y in range(org_image_array.shape[1]) :
                    if(i == 0) :
                        if(org_image_array[x][y] < thereshold_array[i]) :
                            object_detection_img[x][y] = i
                    else : 
                        if(org_image_array[x][y] < thereshold_array[i] and org_image_array[x][y] > thereshold_array[i-1]) :
                            object_detection_img[x][y] = i
                            #print("here")
                    if(org_image_array[x][y] > thereshold_array[n-1]) :
                        object_detection_img[x][y] = n    
        new_means_array = recalculate_means(org_image_array, object_detection_img, n)
        print(new_means_array) 
        if(np.array_equal(means_array,new_means_array)) :
            means_changed = False
        else :
            means_array = new_means_array
            thereshold_array = recalculate_theresholds(means_array)
    
    segmented_image_array = generate_gray_segmented_image(object_detection_img, thereshold_array)
    segmented_image = Image.fromarray(segmented_image_array)
    segmented_image = segmented_image.convert("L")
    return segmented_image, thereshold_array
    
image_filepath = "GUC"
org_image = Image.open(image_filepath + ".jpg")
segmented_image, thereshold_array = detect_objects(org_image, 3)
print(thereshold_array)                
segmented_image.save(image_filepath + "_segmented image.jpg")

#detect_objects(None,4)
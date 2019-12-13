# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:20:38 2019

@authors: Ibram Medhat, Basem Rizk
"""

import random
import numpy as np
from PIL import Image

def print_progress(iteration_type, iteration_value, end_value = 0, upper_bound_exist = False):
    if(upper_bound_exist):
        iteration_value = np.around((iteration_value/end_value)*100,
                                    decimals = 1)
    print( '\r ' + iteration_type + ' %s' % (str(iteration_value)),
              end = '\r')

#Calculates the means of the different segments in the image based on the theresholds
#and pixels in each segement
def recalculate_means(org_image_array, object_detection_img, n):
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
#segmented and n. The function should return the computed n thresholds, 
# a binary image for each segment
#and one segmented gray-scale image with each segment assigned a different gray-level
def detect_objects(org_image_array, n) :
    def random_thresholds(num_of_thresholds):
        #Using Random initialization for n theresholds
        threshold_array = np.zeros((num_of_thresholds,))
        for i in range(num_of_thresholds) :
            threshold_array[i] = random.randint(0,255)
        return np.sort(threshold_array)
 
    means_array = np.zeros(n+1)
    threshold_array = random_thresholds(n)
    print("Init thresholds: " + str(threshold_array))
    object_detection_img =\
        np.zeros(org_image_array.shape[0]*\
                 org_image_array.shape[1]).reshape(org_image_array.shape[0],
                                      org_image_array.shape[1])
    
    means_changed = True
    
    iteration = 1
    while(means_changed) :
        print_progress("iteration: ", iteration)
        iteration+=1
        
        #grouping the orginal image into different groups according to the theresholds values
        for i in range(n) :
            for x in range(org_image_array.shape[0]) :
                for y in range(org_image_array.shape[1]) :
                    if(i == 0) :
                        if(org_image_array[x][y] < threshold_array[i]) :
                            object_detection_img[x][y] = i
                    else : 
                        if(org_image_array[x][y] < threshold_array[i]\
                           and org_image_array[x][y] > threshold_array[i-1]) :
                            object_detection_img[x][y] = i
                            #print("here")
                    if(org_image_array[x][y] > threshold_array[n-1]) :
                        object_detection_img[x][y] = n    
                        
        new_means_array = recalculate_means(org_image_array, object_detection_img, n)
#        print(new_means_array) 
        
        if(np.array_equal(means_array,new_means_array)) :
            means_changed = False
        else :
            means_array = new_means_array
            threshold_array = recalculate_theresholds(means_array)
    print()
    print("Converged.")
    print("Final thresholds: " + str(threshold_array))

    segmented_image_array = generate_gray_segmented_image(object_detection_img,
                                                          threshold_array)
    segmented_image = Image.fromarray(segmented_image_array)
    segmented_image = segmented_image.convert("L")
    return segmented_image, threshold_array
   
    
def extract_objects(org_img_array, threshold_array):
    all_objects = np.zeros((threshold_array.shape[0] + 1,
                            org_img_array.shape[0],
                            org_img_array.shape[1]))
    all_images = []
    full_threshold_array =\
        np.concatenate((np.array([0]), threshold_array, np.array([255])))
    for i in range(full_threshold_array.shape[0] - 1):    
        all_objects[i] = 255*\
                        ((org_img_array > full_threshold_array[i])*\
                         (org_img_array < full_threshold_array[i+1]))
                
        image = Image.fromarray(all_objects[i])
        image = image.convert("L")
        all_images.append(image)
        
    return all_images, all_objects    
    
# =============================================================================
# ------Application
# =============================================================================
def segment(org_img_array, num_of_thresholds, image_filepath):
    print("Segement using Optimal Thresholding with n = " + str(num_of_thresholds))
    
    # Apply Optimal-thesholding-segementation
    segmented_image, threshold_array = detect_objects(org_img_array, num_of_thresholds)
    segmented_image.save(image_filepath + "_" + str(num_of_thresholds) + ".jpg")
    
    with open("Thresholds_" + str(num_of_thresholds) + ".txt",  "w") as f:
        f.write(str(threshold_array))
        
    # Extract binary images based on computed segmentations
    all_images, all_objects = extract_objects(org_img_array, threshold_array)
    for i in range(len(all_images)):
        all_images[i]\
            .save(image_filepath + "_" + str(num_of_thresholds) + "_" + str(i+1) + ".jpg")
        print("Saved image of object " + str(i+1))
    print("Done.")
    
    
image_filepath = "GUC"
org_image = Image.open(image_filepath + ".jpg")
org_img_array = np.array(org_image)

segment(org_img_array, 3, image_filepath)
segment(org_img_array, 4, image_filepath)
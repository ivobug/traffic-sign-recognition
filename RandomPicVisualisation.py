# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 19:57:43 2022

@author: Ivan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

templatesPath='source/templates'
testPath='source/images'

testImages=os.listdir(testPath)
#Getting random image from folder
randomImage=testImages[random.randint(0,len(testImages)-1)]
classNames=[]
templateImages=[]
myList=os.listdir(templatesPath)

#Defining RGB min-max color values choosen by experimenting
lower_blue = np.array([55,130,80])
upper_blue = np.array([255,255,255])
#initialization SIFT feature matching
SIFT=cv2.SIFT_create()

#Read template images by OpenCV and add them and their names to lists 
for cl in myList:
    imgCur= cv2.imread(f'{templatesPath}/{cl}',0)
    templateImages.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
    
    
# detect and compute template images Descriptors and store them 
def findDes(images):
    desList=[]
    for image in images:
        kp,des=SIFT.detectAndCompute(image, None)
        desList.append(des)
    return desList

desList=findDes(templateImages)

# Compute Descriptors for random image and find matches for both of 
# templet images descriptors with ratio test 0.75
def findID(img, desList, thres=0):
    kp2,des2=SIFT.detectAndCompute(img, None)
    bf= cv2.BFMatcher()
    matchList=[]
    #set at -1 because 'bike' has value 0
    finalVal=-1
    try:
        for des in desList:
            matches=bf.knnMatch(des,des2,k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    #print(matchList)
    #take biggest value in matchList and return it as a function output
    if len(matchList)!=0:
        if max(matchList)>thres:
            finalVal=matchList.index(max(matchList))
            
    return finalVal


img = cv2.imread(f'{testPath}/{randomImage}')
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width= img.shape[:2]

#Creating ROI on HSV by applying mask to each layer
ROI= np.array([[(0,0),(width,0),(width,150),(0,150)]], dtype= np.int32)

for i in range(3):    
    blank= np.zeros_like(im_hsv[:,:,i])
    
    region_of_interest= cv2.fillPoly(blank, ROI,255)
    region_of_interest_image= cv2.bitwise_and(im_hsv[:,:,i], region_of_interest)
            
    im_hsv[:,:,i]=region_of_interest_image

#getting the mask image from the HSV image using threshold values
mask = cv2.inRange(im_hsv, lower_blue, upper_blue )

#extracting the contours of the object
contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#sorting the contour based of area
contours = sorted(contours, key=cv2.contourArea, reverse=True)

if contours:
    #if any contours are found we take the biggest contour and get bounding box
    (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
    #drawing a rectangle around the object with 15 as margin
    if(box_width>5):
        cv2.rectangle(im_rgb, (x_min - 15, y_min -15),
                      (x_min + box_width + 15, y_min + box_height + 15),(0,255,0), 4)
        # We are doing SIFT feature matching for rectangle with margins, so we have to
        #make sure if rectangle are next to borders
        if(y_min>15 and (x_min>15)and((x_min+box_width+15)<width)):
            classId=findID(im_gray[y_min-15:y_min +box_height+15,x_min-15:x_min +box_width+15],desList)
        else:
            classId=findID(im_gray[y_min:y_min +box_height+20,x_min-20:x_min +box_width],desList)
            
        if classId !=-1:
            print('Detected traffic sign:', classNames[classId])  
        else:
            print('Traffic sign is not detected')

    
plt.imshow(im_rgb)
plt.axis('off')
plt.show()


    
    






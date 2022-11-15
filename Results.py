# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 21:48:56 2022

@author: Ivan
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from sklearn import metrics


templatesPath='source/templates'
testPath='source/images'

testImages=os.listdir(testPath)
randomImage=testImages[random.randint(0,len(testImages)-1)]
print(randomImage)
classNames=[]
templateImages=[]
myList=os.listdir(templatesPath)

lower_blue = np.array([55,130,80])
upper_blue = np.array([255,255,255])

SIFT=cv2.SIFT_create()


linesArr=[]
with open('source/groundtruth.txt') as f:
    lines = f.readlines()
    linesArr.append(lines)

pedesTrue=[]
bikeTrue=[]
pedesPred=[]
bikePred=[]

for line in linesArr[0][1:]:
    separated=line.split(';')
    bikeTrue.append(int(separated[1]))
    pedesTrue.append(int(separated[2][:-1]))
    

for cl in myList:
    imgCur= cv2.imread(f'{templatesPath}/{cl}',0)
    templateImages.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
    
    
    
    
def findDes(images):
    desList=[]
    for image in images:
        kp,des=SIFT.detectAndCompute(image, None)
        desList.append(des)
    return desList

desList=findDes(templateImages)


def findID(img, desList, thres=0):
    kp2,des2=SIFT.detectAndCompute(img, None)
    bf= cv2.BFMatcher()
    matchList=[]
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
    if len(matchList)!=0:
        if max(matchList)>thres:
            finalVal=matchList.index(max(matchList))
            
    return finalVal


for filename in os.listdir(testPath):
    f = os.path.join(testPath, filename)
    if os.path.isfile(f):
        img = cv2.imread(f)
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        height, width= img.shape[:2]
        image_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ROI= np.array([[(0,0),(width,0),(width,150),(0,150)]], dtype= np.int32)
        
        for i in range(3):    
            blank= np.zeros_like(im_hsv[:,:,i])
            
            region_of_interest= cv2.fillPoly(blank, ROI,255)
            region_of_interest_image= cv2.bitwise_and(im_hsv[:,:,i], region_of_interest)
                    
            im_hsv[:,:,i]=region_of_interest_image

        
        mask = cv2.inRange(im_hsv, lower_blue, upper_blue )


        contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if contours:
            (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])
            cv2.rectangle(im_rgb, (x_min - 15, y_min -15),
                          (x_min + box_width + 15, y_min + box_height + 15),(0,255,0), 4)
            if(y_min>15 and (x_min>15)and((x_min+box_width+15)<width)):
                classId=findID(image_gray[y_min-15:y_min +box_height+15,x_min-15:x_min +box_width+15],desList)
            else:
                classId=findID(image_gray[y_min:y_min +box_height+20,x_min-20:x_min +box_width],desList)
                
            if classId == 0:
                bikePred.append(1)
            else:
                bikePred.append(0)
                
            if classId == 1:
                pedesPred.append(1)
            else:
                pedesPred.append(0)



confusion_matrix = metrics.confusion_matrix(bikeTrue, bikePred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.title("Confusion matrix - bike")

confusion_matrix = metrics.confusion_matrix(pedesTrue, pedesPred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.title("Confusion matrix - pedestrian")


plt.show()













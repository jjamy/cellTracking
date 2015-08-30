#!/usr/bin/env python
#---------------------------------------
# Analysis of Neutrophils, NETs and ROS production with Fungus
# Author: Julianne
#---------------------------------------

### IMPORT ALL THE THINGS
%matplotlib inline

import cv2 as cv2
from matplotlib import pyplot as plt
import numpy as np
import time as t
print "OpenCV Version : %s " % cv2.__version__

### ASSIGN VARIABLES
cellThresh = 0
netThresh = 0
redThresh = 0
cells = []
nets = []
ros = []
image_num = 0
#frames = []
video = 'battledome.avi'

##VIDEO PROCESSING
cap = cv2.VideoCapture(video)
cap.open(video)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frames.append(frame)
    cv2.waitKey(1000)

#for frame in frames:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    blur = cv2.medianBlur(frame,5)
    blue_grey = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    bg = cv2.imshow("blue grey",blue_grey)
    plt.imshow(bg)
    plt.show()
    
    ##CELL COUNT
    #Neut mask
    blue_lower = np.array([0,150,150])
    blue_upper = np.array([20,255,255])
    blue_mask = cv2.inRange(hsv,blue_lower, blue_upper)
    blue_filtered = cv2.bitwise_and(frame,frame,mask=blue_mask)
    
    cimg = cv2.cvtColor(blue_grey,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(blue_grey,cv2.cv.CV_HOUGH_GRADIENT,1,3,param1=10,param2=10,minRadius=0,maxRadius=8)

    if circles is None:
        pr = cv2.imshow("preview", frame)
        plt.imshow(pr)
        plt.show()
        continue
    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
       #print i
       cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),1) # draw the outer circle
       cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle

    pr = cv2.imshow("preview", frame)
    plt.imshow(pr)
    plt.show()
    print "Circles detected: ",len(circles[0])
    
    ##COLOR AREA
    ##ROS
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    red_grey = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    rg = cv2.imshow("blue grey",blue_grey)
    plt.imshow(rg)
    plt.show()
    
    #ROS Mask
    red_lower = np.array([120,60,30])
    red_upper = np.array([180,100,70])
    red_mask = cv2.inRange(hsv,red_lower,red_upper)
    red_filtered = cv2.bitwise_and(frame,frame,mask=red_mask)
    
    contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 200<cv2.contourArea(cnt)<5000:
            cv2.drawContours(red_mask,[cnt],0,(0,255,0),2)
            cv2.drawContours(red_mask,[cnt],0,255,-1)
            
    ##NETS
    net_blur = cv2.medianBlur(frame,5)
    net_grey = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    ng = cv2.imshow("net grey",net_grey)
    plt.imshow(ng)
    plt.show()
    
    #NETS Mask
    blue2_lower = np.array([0,150,0])
    blue2_upper = np.array([20,255,150])
    blue2_mask = cv2.inRange(hsv,blue2_lower, blue2_upper)
    blue2_filtered = cv2.bitwise_and(frame,frame,mask=blue2_mask)
    bm = cv2.imshow("blue mask",blue2_mask)
    plt.imshow(bm)
    plt.show()
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,127,255,0)
    gray2 = gray.copy()
    mask = np.zeros(gray.shape,np.uint8)
    
    contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 200<cv2.contourArea(cnt)<5000:
            cv2.drawContours(frame,[cnt],0,(0,255,0),2)
            cv2.drawContours(mask,[cnt],0,255,-1)
            
        # with open('cell_count.csv', 'w') as csvfile:
    #     fieldnames = ["image","count"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writerow({'image': image_num, 'count': len(circles[0])})
    image_num += 1


##!/usr/bin/env python
##---------------------------------------
## Analysis of Neutrophils, NETs and ROS production with Fungus
## Author: Julianne
##---------------------------------------
#
#### IMPORT ALL THE THINGS
#import cv2
#import csv
#import numpy as np
#
#### ASSIGN VARIABLES
#cellThresh = 0
#netThresh = 0
#redThresh = 0
#cells = []
#nets = []
#ros = []
#image_num = 0
#video = 'battledome.avi'
#
#cap = cv2.VideoCapture(video)
#cap.open(video)
#
#while(cap.isOpened()):
#    # Capture frame-by-frame
#    ret, frame = cap.read()
#    cv2.waitKey(1000)
#    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
#
#    blur = cv2.medianBlur(frame,5)
#    blue_grey = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
#    cv2.imshow("blue grey",blue_grey)
#    
#    #Neut mask
#    blue_lower = np.array([0,150,150])
#    blue_upper = np.array([20,255,255])
#    blue_mask = cv2.inRange(hsv,blue_lower, blue_upper)
#    blue_filtered = cv2.bitwise_and(frame,frame,mask=blue_mask)
#    
#    cimg = cv2.cvtColor(blue_grey,cv2.COLOR_GRAY2BGR)
#    circles = cv2.HoughCircles(blue_grey,cv2.cv.CV_HOUGH_GRADIENT,1,3,param1=10,param2=10,minRadius=0,maxRadius=8)
#
#    if circles is None:
#        cv2.imshow("preview", frame)
#        continue
#    circles = np.uint16(np.around(circles))
#
#    for i in circles[0,:]:
#       #print i
#       cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),1) # draw the outer circle
#       cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle
#
#    cv2.imshow("preview", frame)
#    print "Circles detected: ",len(circles[0])
#    
#    ##ROS
#    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
#    red_grey = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
#    cv2.imshow("blue grey",blue_grey)
#    
#    #ROS Mask
#    red_lower = np.array([120,60,30])
#    red_upper = np.array([180,100,70])
#    red_mask = cv2.inRange(hsv,red_lower,red_upper)
#    red_filtered = cv2.bitwise_and(frame,frame,mask=red_mask)
#    
#    contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#    for cnt in contours:
#        if 200<cv2.contourArea(cnt)<5000:
#            cv2.drawContours(red_mask,[cnt],0,(0,255,0),2)
#            cv2.drawContours(red_mask,[cnt],0,255,-1)
#    ##NETS
#    red_blur = cv2.medianBlur(frame,5)
#    red_grey = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
#    cv2.imshow("blue grey",blue_grey)
#    
#    #NETS Mask
#    blue2_lower = np.array([0,150,0])
#    blue2_upper = np.array([20,255,150])
#    blue2_mask = cv2.inRange(hsv,blue2_lower, blue2_upper)
#    blue2_filtered = cv2.bitwise_and(frame,frame,mask=blue2_mask)
#    cv2.imshow("blue grey",blue_grey)
#
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    ret,gray = cv2.threshold(gray,127,255,0)
#    gray2 = gray.copy()
#    mask = np.zeros(gray.shape,np.uint8)
#    
#    contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#    for cnt in contours:
#        if 200<cv2.contourArea(cnt)<5000:
#            cv2.drawContours(frame,[cnt],0,(0,255,0),2)
#            cv2.drawContours(mask,[cnt],0,255,-1)
#            
#        # with open('cell_count.csv', 'w') as csvfile:
#    #     fieldnames = ["image","count"]
#    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#    #     writer.writerow({'image': image_num, 'count': len(circles[0])})
#    image_num += 1

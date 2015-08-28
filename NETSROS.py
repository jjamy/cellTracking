#---------------------------------------
# Analysis of Neutrophils, NETs and ROS production with Fungus
# Author: Julianne
#---------------------------------------

### IMPORT ALL THE THINGS
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

### ASSIGN VARIABLES
X = 'F:\Imaging\BATTLEDOME AVI\20150804 Neuts DPI NEi BattleDome + CellROX005.avi' #Video 
cap = cv2.VideoCapture('F:\Imaging\BATTLEDOME AVI\20150804 Neuts DPI NEi BattleDome + CellROX005.avi')
cellThresh = 0
netThresh = 0
redThresh = 0
cells = []
nets = []
ros = []

### CODE
#video processing
fgbg = cv2.createBackgroundSubtractorMOG2()
## Video to image
#while True:
 #   if cap.grab():
#        flag, frame = cap.retrieve()
   #     cv2.imshow('video', frame)
cap = cv2.VideoCapture()
cap.open('F:\Imaging\BATTLEDOME AVI\20150804 Neuts DPI NEi BattleDome + CellROX005.avi')
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(ret)
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    #mfgmask = fgbg.apply(frame)

    cv2.imshow('frame',frame)
    cv2.waitKey(1000)
    cv2.imshow('frame',fgmask)
    cv2.waitKey(1000)
    
    ## Code RGB to HSV
    cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    thb1l = np.array([110,50,50]) #Nets
    thb1h = np.array([130,200,100])
    thb2l = np.array([140,220,110]) #Blue Neuc: 158,240,120
    thb2h = np.array([160,255,255])
    thrh = np.array([180,150,60]) #Ros
    thrl = np.array([255,175,80])
    
#     #threshold for blue and red and calculate are of each
#     ret,th1 = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
#     th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
#     th3 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
#     titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
#     images = [frame, th1, th2, th3]

#     for i in xrange(4):
#         plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#         plt.title(titles[i])
#         plt.xticks([]),plt.yticks([])
#     plt.show()
    
    #edges
    edges = cv2.Canny(cap,100,200)
    plt.subplot(121),plt.imshow(cap,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()
    
     ##circle detection for cells
    cimg = cv2.cvtColor(cap,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(cap,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('detected circles',cimg)
    #cv2.waitKey(0)
    
    ##trace image
    cells = cells.append(cellArea) #area of single cell
    nets = nets.append(netArea/cells)
    ros = ros.append(redArea/cells)
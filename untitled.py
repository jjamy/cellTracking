#!/usr/bin/env python
#---------------------------------------
# Analysis of Neutrophils
# Author: Julianne
#---------------------------------------

import cv2 as cv2
from matplotlib import pyplot as plt
import numpy as np
import time as t
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp
import trackpy.predict
import cPickle
import ipdb
import sys
import os
# import av
from tqdm import tqdm, trange

print "OpenCV Version : %s " % cv2.__version__

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException, "Timed out!"
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

### ASSIGN VARIABLES
cellThresh = 0
netThresh = 0
redThresh = 0
cells = []
image_num = 0
history = 10
frames = []


#video = os.path.realpath("/media/julianne/Passport/LEO/114/FI32a.avi")
#video = os.path.realpath("/media/luke/Passport/LEO/114/FI32a.avi")

#video = os.path.realpath("/home/luke/Downloads/1a.avi")
#prefix = "/media/luke/Passport/LEO/114"
prefix = sys.argv[1]
videos = [os.path.join(prefix, p) for p in os.listdir(prefix)]
# video = sys.argv[1]
# videos = [os.path.join(video, s) for s in os.listdir(video)]
# videos = sorted(videos)
# videos = videos[0:24]

def half_show(name, frame):
    cv2.imshow(name, cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2)))

def pixel2micron(area):
    # pixels**2  * (microns* / pixels)**2 = microns**2
    area = area*((50./109)**2)
    return area

def calculate_cell_count(frame):
    # print "calculating"
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    blur = cv2.medianBlur(frame,7)
    # blurdst = cv2.fastNlMeansDenoising(blur,10,10,7,21)
    blurdst = cv2.fastNlMeansDenoising(blur,h=10,templateWindowSize=5,searchWindowSize=11)


    # blurdst2 = cv2.fastNlMeansDenoising(blur,h=10,templateWindowSize=7,searchWindowSize=21)
    # half_show('blurdst', blurdst)
    # half_show('blurdst2', blurdst2)
    # half_show('blur', blur)

    # cv2.waitKey(0)
    grey = cv2.cvtColor(blurdst, cv2.COLOR_RGB2GRAY);
    circles = cv2.HoughCircles(grey,cv2.cv.CV_HOUGH_GRADIENT,1,20,param1=15,param2=5,minRadius=2,maxRadius=10)

    if circles is None:
        return (0,None)

    filteredcircles = []

    for i in circles[0,:]:
        if i[0]>150:
            filteredcircles.append(i)

    for i in filteredcircles:
       cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),3) # draw the outer circle
       cv2.circle(blurdst,(i[0],i[1]),i[2],(0,255,0),1) # draw the outer circle
       cv2.circle(blurdst,(i[0],i[1]),2,(0,0,255),3) # draw the center of the circle

    numCells = len(filteredcircles)
    # half_show("preview", frame)
    # half_show("preview-blue", blurdst)
    # cv2.waitKey(100)
    # print "circles: ", filteredcircles
    return (numCells,filteredcircles)

def tracking(video):
    # print "running tracking"
    pimsframes = pims.Video(video, as_grey = True)
    fgbg = cv2.BackgroundSubtractorMOG()
    framesmask = []
    framecount = 0
    blurredframes = []
    # HACK to flip if has space in name. #TODO get all videos correctly alligned...
    if " " in video:
        pimsframes = [p[:, ::-1] for p in pimsframes]

    pimsframes = [frame[:,400:] for frame in pimsframes]
    for frame in pimsframes:
        # frame = cv2.GaussianBlur(frame,(9,9),0)
        # frame = cv2.medianBlur(frame, 7)
        # if align remove
        frame = cv2.GaussianBlur(frame,(11,11),7)
        frame = cv2.medianBlur(frame, 3)
        fgmask = fgbg.apply(frame, learningRate=1.0/history)
        framesmask.append(fgmask)
        framecount += 1
        blurredframes.append(frame)

    background_sub = [m * frame for m,frame in zip(framesmask, pimsframes)]
    if False:
        for i in range(100):
            cv2.imshow("asdf", background_sub[i])
            cv2.imshow("mask", framesmask[i])
            cv2.imshow("orig", pimsframes[i])
            cv2.imshow("blur", blurredframes[i])
            cv2.waitKey(0)
    # for i, f in enumerate(framesmask):
    #     half_show("asdf", f)
    #     cv2.waitKey(0)
    #     print i
    cells = []
    track = []

    to_track = background_sub
    minmass = 3000

    f = tp.batch(to_track[:], 11, minmass=minmass, invert=False, noise_size=3)
    # for j in range(20,100):
    #     f = tp.locate(to_track[j], 11, invert=False, minmass = minmass, noise_size=3)
    #     plt.figure(1)
    #     tp.annotate(f, to_track[j])
    #     plt.show()
    # ipdb.set_trace()
    print "linking"
    try:
        # t = tp.link_df(f, 100, memory=3)
        t = tp.link_df(f, 100, memory=1)

    except Exception:
        print "FAILED on", video
        return None
    print "done"
    # plt.figure(2)
    # tp.plot_traj(t)
    # plt.show()
    return t

def heatmap(circles,image):
    # print "running heatmap"
    #max values: [179,255,255]
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    hsv[:,:,0] = 51
    hsv[:,:,1] = 235
    offset = 400

    res = 3
    #jcount = np.zeros((hsv.shape[0] * res + offset*2, hsv.shape[1] * res + offset*2))
    count = np.zeros((hsv.shape[1] * res + offset*2, hsv.shape[0] * res + offset*2))
    s = count.shape

    r = 10
    circ_temp = np.zeros((r*2*res,r*2*res))
    for i in range(r*2*res):
        for j in range(r*2*res):
            dx = (i - (r*res))
            dy = (j - (r*res))
            dist = np.sqrt(dx**2 + dy**2)
            if dist < r*res:
                circ_temp[i,j] = 1

    #circ_temp = gkern(r*2*res, nsig=1)
    #plt.imshow(circ_temp)
    #plt.show()

    for i, point in tqdm(list(enumerate(circles))):
        tl = (max(0, point[0]-r),max(point[1]-r, 0)) #make sure on min or max
        br = (min(s[0], point[0]+r),min(s[1], point[1]+r)) #make sure on min or max
        tl = [int(x) for x in tl]
        br = [int(x) for x in br]

        dx = tl[0] - br[0]
        dy = tl[1] - br[1]
        #count[tl[0]:br[0], tl[1]:br[1]] += 1
        try:
            count[tl[0]*res+offset:br[0]*res+offset, tl[1]*res+offset:br[1]*res+offset] += circ_temp
            # count[tl[0]*res+offset:br[0]*res+offset, tl[1]*res+offset:br[1]*res+offset] += 1
        except:
            print "error"

    plt.imshow(count)
    plt.colorbar()
    plt.show()

    return count


def process_video(video):
    print "working on video:", video
    cap = cv2.VideoCapture(video)
    cap.open(video)
    cur_frame_idx = 0
    redareas = []
    netareas = []
    cellcounts = []
    cellareacounts = []
    circs = []
    height = 588
    width = 1600
    blank = np.zeros((height, width, 3), np.uint8) #change for size of video
    fgbg = cv2.BackgroundSubtractorMOG()

    # if True:
    #     rets, frames = [], []
    #     # mask_vid = []
    #     while(cap.isOpened()):
    #         # Capture frame-by-frame
    #         ret, frame = cap.read()
    #         if frame is None:
    #             break
    #         rets.append(ret)
    #         frames.append(frame)
    #     # XXXXX REMOVE ME
    #     frames = frames[50:70]
    #     for ret, frame in tqdm(list(zip(rets, frames))):
    #         fgmask = fgbg.apply(frame, learningRate=1.0/history)
    #         mask_rbg = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)
    #         # half_show('frame', frame)
    #         # half_show('background subtractor',fgmask)
    #         # cv2.waitKey(100)
    #         cellcount,circ = calculate_cell_count(mask_rbg)
    #         if circ != None:
    #             for value in circ:
    #                 circs.append(value)
    #         cellcounts.append(cellcount)

    #         # mask_vid += mask_rbg

    #     cPickle.dump(circs, open("circs.pkl", "w"))

    # circs = cPickle.load(open("circs.pkl"))

    # hm = heatmap(circs, blank)
    # cv2.imshow('heatmap', hm)
    # cv2.waitKey(1000)

    df = tracking(video)
    if df is None:
        print "FAILED ON", video
        return
    output = "out/"
    name = video.split("/")[-1].split(".")[-2]
    df.to_csv("output/%s.csv"%name)
    # times = np.linspace(0, 6, len(video))
    # cellcounts = np.array(cellcounts)
    # times = np.array(times)
    # mask = np.arange(0, len(cellcounts))[10:]

    # plt.plot(times, cellcounts[mask])
    # plt.figure()
    # plt.plot(times, cellareacounts[mask])
    # plt.show()

    # frame = pd.DataFrame()
    # frame['times'] = times
    # frame['cell_count'] = cellcounts
    # frame['cell_count_area'] = cellareacounts

    # import os
    # base_name = os.path.basename(video)
    # frame.to_csv(base_name+'_export.csv', indexGG=False)

# for v in tqdm(videos):
import multiprocessing
pool = multiprocessing.Pool(8)
import numpy as np
np.random.shuffle(videos)
videos = [v for v in videos if ".avi" in v]

path = lambda video: "output/"+video.split("/")[-1].split(".")[-2] + ".csv"

videos = [v for v in videos if not os.path.exists(path(v))]
print len(videos)
print videos

#for v in tqdm(videos):
#process_video(v)

    # try:
    #     with time_limit(10):
    #         process_video(v)
    # except TimeoutException, msg:
    #     print "Timed out! on", v

# map(process_video, videos)
pool.map(process_video, videos)
#process_video(video)

# blank = np.zeros((550, 1200, 3), np.uint8)
# hm = heatmap([[250,250],[300,300],[250,250],[300,300],[250,250],[300,300],[250,250],[300,300],[250,250],[300,300],[250,250],[300,300],[250,250],[300,300],[250,250],[300,300],[250,250],[300,300],[250,250],[300,300],[250,250],[300,300]],blank)
# cv2.imshow('heatmap', hm)
# cv2.waitKey(5000)
# tracking(video)
#        # with open('cell_count.csv', 'w') as csvfile:
#    #     fieldnames = ["image","count"]
#    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#    #     writer.writerow({'image': image_num, 'count': len(circles[0])})
#    image_num += 1

import cv2 as cv
import numpy as np
import random as rng
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt

# To find convex hull
def con_hull(points):  
    hull = ConvexHull(points)
    return hull.simplices

# Function to find CArea
def PolyArea2D(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area
#def roi(img,degree):
# Reading two frames

img1 = cv.imread('frame0.jpg')  
img2 = cv.imread('frame0.jpg')

def collisionfunc(img1,img2):

# convert images to grayscale
    gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray_img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    #gray_img1=cv.Canny(gray_img1,100,200)
    #gray_img2=cv.Canny(gray_img2,100,200)

    # create SIFT object
    sift = cv.SIFT_create()

    # detect SIFT features in both images
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray_img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray_img2,None)
    # create feature matcher
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

    # match descriptors of both images
    matches = bf.match(descriptors_1,descriptors_2)
    # sort matches by distance

            


    # Initialize lists which store x, y coordinates of keypoints
    list_kp1 = []
    list_kp2 = []

    # Initialize lists which store size of keypoints
    s_kp1=[]
    s_kp2=[]
    newmatches=[]
    # For each match, filling list for x,y coordinates and size of keypoints
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = keypoints_1[img1_idx].pt
        (x2, y2) = keypoints_2[img2_idx].pt
        if((((x2)**2+(y2)**2)/((x1)**2+(y1)**2))**0.5<=0.7):
            newmatches.append(mat)
            s1= keypoints_1[img1_idx].size
            s2= keypoints_2[img2_idx].size

            # Append to each list
            list_kp1.append((x1, y1))
            list_kp2.append((x2, y2))
            s_kp1.append(s1)
            s_kp2.append(s2)

    # Filtering keypoints in frame 2 based on size(mpk2[i]) > size(mkp1[i])
    collision=False
    if(len(list_kp1)>3):
        mkp1=[]
        mkp2=[]
        mkps1=[]
        mkps2=[]
        for i in range(0,len(s_kp1)):
            if(s_kp1[i]<s_kp2[i]):
                mkp2.append(list_kp2[i])
                mkp1.append(list_kp1[i])
                mkps1.append(s_kp1[i])
                mkps2.append(s_kp2[i])


        # Final numpy list of keypoints
        kp1 = np.array(mkp1)
        kp2 = np.array(mkp2)

        # Finding convex hull
        C1=con_hull(kp1)
        C2=con_hull(kp2)


        # Calculating CArea1 and CArea2
        Carea1=PolyArea2D(C1)
        Carea2=PolyArea2D(C2)

        ratioC=Carea2/Carea1
        ratio_mkp=0
        for i in range(len(mkps1)):
            ratio_mkp+=mkps2[i]/mkps1[i]
        ratio_mkp=ratio_mkp/len(mkps1)

        # Calculating ratio for final analysis
        if(ratioC>1.7 or ratio_mkp>1.2):
            collision=True

    print(collision)

collisionfunc(img1,img2)


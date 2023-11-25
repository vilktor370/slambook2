import cv2 as cv
from scipy.optimize import minimize
import numpy as np

def orb(img):
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des



img1 = cv.imread('ch7/1.png')
img2 = cv.imread('ch7/2.png')
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)

# find the keypoints and descriptors with ORB
orb = cv.ORB_create()
keypoint1, descriptor1 = orb.detectAndCompute(img1,None)
keypoint2, descriptor2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(descriptor1,descriptor2)
max_distance = max([d for d in matches])
print(matches)
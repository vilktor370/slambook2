import cv2
import numpy as np
from scipy.optimize import minimize
"""
This is an implementation of ICP algorithm.

THis algorithm solves the 3D - 3D problem.

"""
def feature_match(img1, img2):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    keypoint1, descriptor1 = orb.detectAndCompute(img1,None)
    keypoint2, descriptor2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(descriptor1,descriptor2)
    res = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1,keypoint1,img2,keypoint2,res,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("image", img3)
    cv2.waitKey(0)
    return res, keypoint1, keypoint2

# convert [u, v] -> [u, v, 1], [x, y, z] -> [x, y, z, 1]
def homogenous(pts):
    ones = np.ones((pts.shape[0], 1))
    return np.hstack([pts, ones])

# convert [u ,v] to [x, y]
def pixel2cam(u ,v, K):
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return x, y

color_img1 = cv2.imread("1.png")
color_img2 = cv2.imread("2.png")
depth_img1 = cv2.imread("1_depth.png", 0)
depth_img2 = cv2.imread("2_depth.png", 0)
if color_img1 is None or color_img2 is None or depth_img1 is None or depth_img2 is None:
    print("Image not read succesful!")
    exit(1)
    
matches, keypoint1, keypoint2 = feature_match(img1=color_img1, img2=color_img2)
K = np.array([
    [520.9, 0, 325.1],
    [0, 521, 249.7],
    [0, 0, 1.0]
])

img1_xyz = []
img2_xyz = []
for i, m in enumerate(matches):
    v1 = int(keypoint1[m.queryIdx].pt[1])
    u1 = int(keypoint1[m.queryIdx].pt[0])
    v2 = int(keypoint2[m.trainIdx].pt[1])
    u2 = int(keypoint2[m.trainIdx].pt[0])
    d1 = depth_img1[v1, u1]
    d2 = depth_img2[v2, u2]
    if d1 != 0 and d2 != 0:
        dd1 = d1 / 5000
        dd2 = d2 / 5000
        x1, y1 = pixel2cam(u1, v1, K)
        x2, y2 = pixel2cam(u2, v2, K)
        img1_xyz.append(np.array([x1 * dd1, y1 * dd1, dd1]))
        img2_xyz.append(np.array([x2 * dd2, y2 * dd2, dd2]))
img1_xyz = np.array(img1_xyz)
img2_xyz = np.array(img2_xyz)

# SVD method
q1 = img1_xyz - np.mean(img1_xyz)
q2 = img2_xyz - np.mean(img2_xyz)
N = q1.shape[0]
W = np.zeros((3,3))
for i in range(N):
    W += q1[i, :][:, None].dot(q2[i, :][None, :])
U,S,V = np.linalg.svd(W)
R = U.dot(V.T)
if np.linalg.det(R) < 0.0:
    R = -1 * R
t = img1_xyz - R.dot(img2_xyz.T).T
fit_error = np.linalg.norm(img1_xyz - (R.dot(img2_xyz.T).T + t))
print("ICP reprojection error:", fit_error)
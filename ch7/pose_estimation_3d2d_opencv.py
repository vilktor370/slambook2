import cv2
import numpy as np
from scipy.optimize import minimize

def feature_match(img1, img2):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    keypoint1, descriptor1 = orb.detectAndCompute(img1,None)
    keypoint2, descriptor2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(descriptor1,descriptor2)
    res = sorted(matches, key=lambda x: x.distance)[:4]
    # img3 = cv2.drawMatches(img1,keypoint1,img2,keypoint2,res,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("image", img3)
    # cv2.waitKey(0)
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
img2_uv = []
for i, m in enumerate(matches):
    v = int(keypoint1[m.queryIdx].pt[1])
    u = int(keypoint1[m.queryIdx].pt[0])
    d = depth_img1[v, u]
    if d != 0:
        dd = d / 5000
        x, y = pixel2cam(u, v, K)
        img1_xyz.append(np.array([x * dd, y * dd, dd]))
        img2_uv.append(np.array(keypoint2[m.trainIdx].pt))


# opencv
opencv_xyz = np.array(img1_xyz).astype(np.float32)
opencv_uv = np.array(img2_uv).astype(np.float32)
K = K.astype(np.float32)
stat, r, t = cv2.solvePnP(opencv_xyz,opencv_uv, K, np.zeros((5, 1), dtype=np.float32))
R = cv2.Rodrigues(r)[0]
if stat:
    opencv_P = K.dot(np.hstack([R, t]))
    calc_uv =opencv_P.dot(homogenous(opencv_xyz).T).T
    calc_uv /= (calc_uv[:, -1][:, None])
    calc_uv = calc_uv[:, :-1]
    fit_error = np.linalg.norm(opencv_uv - calc_uv)
    print("Opencv reprojection error:", fit_error)
    
# optimize
def reprojection_error(x):
    x = np.reshape(x, (3,4))
    calc_uv =x.dot(homogenous(opencv_xyz).T).T
    calc_uv /= (calc_uv[:, -1][:, None])
    calc_uv = calc_uv[:, :-1]
    fit_error = np.linalg.norm(opencv_uv - calc_uv)
    return fit_error

res = minimize(reprojection_error, opencv_P)
optimize_P = res.x
fit_error = reprojection_error(optimize_P)
print("Optimize reprojection error:", fit_error)
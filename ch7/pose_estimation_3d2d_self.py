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
    
    # # visualization
    # img3 = cv2.drawMatches(img1,keypoint1,img2,keypoint2,res,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("image", img3)
    # cv2.waitKey(0)
    img1_pts = []
    img2_pts = []
    for m in matches:
        img1_pts.append(keypoint1[m.queryIdx].pt)
        img2_pts.append(keypoint2[m.queryIdx].pt)
    img1_pts_h = homogenous(np.array(img1_pts))
    img2_pts_h = homogenous(np.array(img2_pts))
    return img1_pts_h, img2_pts_h

# convert [u, v] -> [u, v, 1], [x, y, z] -> [x, y, z, 1]
def homogenous(pts):
    ones = np.ones((pts.shape[0], 1))
    return np.hstack([pts, ones])

# convert [u ,v, 1] to [x, y ,z, 1]
def pixel2cam(img_uv_h, depth_img, K):
    xyz = np.linalg.inv(K).dot(img_uv_h.T).T
    new_xyz = []
    for i in range(img1_uv_h.shape[0]):
        d = depth_img[int(img1_uv_h[i, 1]), int(img1_uv_h[i, 0])]
        # if d != 0:
        new_xyz.append([xyz[i, 0], xyz[i, 1], d])
    return np.array(new_xyz)

def decompose_projection_matrix(P):
    # Normalize the projection matrix to improve numerical stability
    P /= P[2, 3]
    
    # Extract the intrinsic matrix (K) and the first two columns of the rotation matrix (R)
    K, R = np.linalg.qr(np.linalg.inv(P[:, :3]))

    # Ensure the diagonal elements of K are positive
    T = np.diag(np.sign(np.diag(K)))
    K = np.dot(K, T)
    R = np.dot(T, R)

    # Extract the translation vector (t)
    t = np.linalg.inv(K) @ P[:, 3]

    return K, R, t


def dlt(img1_xyz_h, img2_uv_h):
    A = []
    N = img1_xyz_h.shape[0]
    for i in range(N):
        u, v = img2_uv_h[i, :-1]
        if i % 2 == 0:
            row = np.hstack([img1_xyz_h[i], np.zeros(4,), -u * img1_xyz_h[i]])
        else:
            row = np.hstack([np.zeros(4,), img1_xyz_h[i], -v * img1_xyz_h[i]])
        A.append(row)
    A = np.array(A)
    
    _, _, V = np.linalg.svd(A)
    T = V[-1]
    
    # normalize
    T /= np.linalg.norm(T)
    
    # verify
    verify_res = A.dot(T)
    tolerance = 0.5
    if np.allclose(verify_res, np.zeros_like(verify_res), atol=tolerance):
        print("SVD answer is correct!")
    return np.reshape(T, (3,4))

def optimize_projection_matrix(img1_xyz_h, img2_uv_h, P):
    reprojection_uv_h = P.dot(img1_xyz_h.T).T
    # reprojection_uv_h /= (reprojection_uv_h[:, -1][:, None])
    reprojection_error_func =lambda x: np.linalg.norm(img2_uv_h - x.reshape((3,4)).dot(img1_xyz_h.T).T)
    result = minimize(reprojection_error_func, P)
    return np.reshape(result.x, (3,4))

color_img1 = cv2.imread("1.png")
color_img2 = cv2.imread("2.png")
depth_img1 = cv2.imread("1_depth.png", 0)
depth_img2 = cv2.imread("2_depth.png", 0)
if color_img1 is None or color_img2 is None or depth_img1 is None or depth_img2 is None:
    print("Image not read succesful!")
    exit(1)
    
img1_uv_h, img2_uv_h = feature_match(img1=color_img1, img2=color_img2)
K = np.array([
    [520.9, 0, 325.1],
    [0, 521, 249.7],
    [0, 0, 1.0]
])

img1_xyz = pixel2cam(img1_uv_h, depth_img1, K)
img1_xyz_h = homogenous(img1_xyz)
# Use DLT to find the initialized value
init_P = dlt(img1_xyz_h, img2_uv_h)
fit_error = np.linalg.norm(img2_uv_h - init_P.dot(img1_xyz_h.T).T)
print("DLT reprojection error:", fit_error)
print("DLT P:\n", init_P)

# use Bundle Adjustment to optimize a better solution
init_P = np.hstack([np.eye(3), np.zeros((3,1))])
P = optimize_projection_matrix(img1_xyz_h, img2_uv_h, init_P)

fit_error = np.linalg.norm(img2_uv_h - P.dot(img1_xyz_h.T).T)
print("After optimization reprojection error:", fit_error)
print("Optimization P:\n", P)

# opencv
opencv_xyz = img1_xyz_h[:, :-1].astype(np.float32)
opencv_uv = img2_uv_h[:, :-1].astype(np.float32)
K = K.astype(np.float32)
stat, r, t = cv2.solvePnP(opencv_xyz,opencv_uv, K, np.zeros((5, 1), dtype=np.float32))
R = cv2.Rodrigues(r)[0]
if stat:
    opencv_P = K.dot(np.hstack([R, t]))
    fit_error = np.linalg.norm(img2_uv_h - opencv_P.dot(img1_xyz_h.T).T)
    print("Opencv reprojection error:", fit_error)
    print("Opencv R:\n", R)
    print("Opencv t:\n", t)
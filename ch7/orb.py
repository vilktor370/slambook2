import cv2 as cv
import numpy as np

def orb(img):
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return kp, des

img1 = cv.imread('1.png')
img2 = cv.imread('2.png')
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
min_distance = min([d.distance for d in matches])

good_match = []
for m in matches:
    if m.distance <= max(2 * min_distance, 30):
        good_match.append(m)
print("Number of matches:", len(good_match))

# filter out useful point for essential matrix, fundemental matrix and homography
point1 = []
point2 = []
for m in matches:
    point2.append(keypoint1[m.trainIdx].pt)
    point1.append(keypoint2[m.queryIdx].pt)
point1 = np.array(point1)
point2 = np.array(point2)

# find fundemental matrix
fundemental_matrix = cv.findFundamentalMat(point1, point2, cv.FM_8POINT)[0]
print("Fundemental matrix:\n", fundemental_matrix)

# find essential matrix
principal_point = (325.1, 249.7)
focal_length = (520.9, 521)
K = np.array([
    [focal_length[0], 0, principal_point[0]],
    [0, focal_length[1], principal_point[1]],
    [0, 0, 1]
])
print("K:\n", K)
essential_matrix = cv.findEssentialMat(point1, point2, K)[0]
print("Essential matrix:\n", essential_matrix)

# find homography
homography_matrix = cv.findHomography(point1, point2, cv.RANSAC, 5)[0]
print("Homography matrix:\n", homography_matrix)
ones = np.ones((point1.shape[0], 1))
point1_h = np.hstack([point1, ones])
point2_h = homography_matrix.dot(point1_h.T).T
point2_h/= (point2_h[:, -1][:, None])
fit_error = np.linalg.norm(point2_h[:, :-1] - point2)
print("Fir error for homography:", fit_error)

# decompose essential matrix
print("--------------------Essential-------------------------")
R1, R2, t = cv.decomposeEssentialMat(essential_matrix)
print("R1:\n", R1)
print("R2:\n", R2)
print('t:\n', t)

# verify essential matrix is correct
t = t[:, 0]
t_hat = np.array([
    [0, -t[2], t[1]],
    [t[2], 0, -t[0]],
    [-t[1], t[0], 0]
])
point1_h = np.hstack([point1, np.ones((point1.shape[0], 1))])
point2_h = np.hstack([point2, np.ones((point2.shape[0], 1))])

x1 = np.linalg.inv(K).dot(point1_h.T)
x2 = np.linalg.inv(K).dot(point2_h.T)
N = x1.shape[1]
for i in range(N):
    X1 = x1.T[i]
    X2 = x2.T[i]
    res = X2.T.dot(t_hat.dot(R1)).dot(X1)
    print(res)

calc_essential_matrix = t_hat.dot(R1)
print('Essential matrix fit error:', np.linalg.norm(essential_matrix - calc_essential_matrix))
calc_fundamental_matrix = np.linalg.inv(K.T).dot(t_hat.dot(R1)).dot(np.linalg.inv(K))
print("Fundamental matrix fit error:",np.linalg.norm(fundemental_matrix - calc_fundamental_matrix) )

# verify homography matrix
for i in matches:
    X1 = np.array([keypoint1[i.queryIdx].pt[0], keypoint1[i.queryIdx].pt[1], 1.0])
    X2 = np.array([keypoint1[i.trainIdx].pt[0], keypoint1[i.trainIdx].pt[1], 1.0])
    calc_X1 = homography_matrix.dot(X2)
    calc_X1 /= calc_X1[-1]
    fit_error = np.linalg.norm(X1[:-1] - calc_X1[:-1])
    print(fit_error)
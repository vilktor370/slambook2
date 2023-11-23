import numpy as np
import cv2

def homogenous(p):
    ones = np.ones((p.shape[0], 1))
    return np.hstack([p, ones])

def normalize(mtx):
    mtx /= mtx[:, -1][:, None]
    return mtx[:, :-1]

XYZ = np.array([
    [10, 10, 10],
    [50, 10, 10],
    [100, 10, 10],
    [100, 100, 100],
    [50, 100, 10],
    [10, 100, 10]
])
uv = np.array([
    [50, 50],
    [75, 50],
    [100, 50],
    [100, 100],
    [75, 100],
    [50, 100]
])

XYZ1 = homogenous(XYZ)
N = XYZ1.shape[0]  # number of correspondeses
A = np.zeros((N*2, N*2))
uv1_vec = uv.reshape(-1)
A[::2, :] = np.hstack([XYZ1, np.zeros((N, 4)), -uv1_vec[::2][:, None] * XYZ1])
A[1::2, :] = np.hstack(
    [np.zeros((N, 4)), XYZ1, -uv1_vec[1::2][:, None] * XYZ1])
print(A)

# Calculate the transformation matrix
U, Sigma, Vt = np.linalg.svd(A)
project_matrix = Vt[-1].reshape(3, 4)
out = cv2.decomposeProjectionMatrix(project_matrix)
k = out[0]
t = (out[2]/out[2][-1])[:-1, :]
R = out[1]
T = k @ np.hstack([R, t])
print("Transformation Matrix:\n", T)

# Fit error
uv_calc = T.dot(XYZ1.T)
uv_calc = normalize(uv_calc.T)
fit_error = np.linalg.norm(uv - uv_calc)
print("Overall fit error:", fit_error)
print("Original 2D points:\n")
print(uv)
print("Reprojected 2D points:\n")
print(uv_calc)

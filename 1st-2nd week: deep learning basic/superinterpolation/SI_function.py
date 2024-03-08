"""

Paper: Super-Interpolation With Edge-Orientation-Based Mapping Kernels for Low Complex 2x Upscaling / Jae-Seok Choi; Munchurl Kim / 09 December 2015

[ Training Phase ]

1. input: external HR images H.
2. Initiallization:
	Generate LR images L from external HR images H
	Generate 3x3 LR patch y and 2x2 patch x pairs.
3. For each LR external patch y, do
4.      For each 2x2 LR sub-patches in y, do
5.		    Apply 2 operators(sh, sv) to compute its edge-direction index
6.	    End For
7.	    Compute EO class index
8.	    Put the current patch y into the corresponding cluster
9. End For
10. For each cluster of the EO class C
11. 	Compute linear mappings matric Mc by using (10)
12. End For
13. Store linear mappings and their EO class index

"""

import numpy as np
import cv2
import math


def SB(image, index):
    kernel_size = 5  # 커널 크기
    blurring = cv2.GaussianBlur(image[index], (kernel_size, kernel_size), 0)
    downscaling = blurring[::2, ::2]
    return downscaling


def make_patch_y(image):
    y = []
    for i in range(image.shape[0]-3+1):
        for j in range(image.shape[1]-3+1):
            patch = image[i:i+3, j:j+3]
            y.append(patch)
    return np.array(y)


def make_patch_x(image):
    x = []
    for i in range(int(image.shape[0]/2)-1):
        for j in range (int(image.shape[1]/2)-1):
            if i != int(image.shape[0]/2)-2:
                if j !=  int(image.shape[1]/2)-2:
                    patch = image[(i+1)*2:(i+1)*2+2, (j+1)*2:(j+1)*2+2]
                    x.append(patch)
    return np.array(x)



sh = [[1, -1], [1, -1]]
sv = [[1, 1], [-1, -1]]

def mP(p):
    gh = np.sum(p*sh)
    gv = np.sum(p*sv)
    mP_value = math.sqrt(gh**2 + gv**2)
    return mP_value

def dP(p):
    gh = np.sum(p*sh)
    gv = np.sum(p*sv)
    dP_value = math.degrees(math.atan(gh/gv))
    if dP_value <= -22.5:
        dP_value = dP_value+180
    elif dP_value > 180:
        dP_value = dP_value-180
    if -22.5 <= dP_value < 22.5:
        return 1
    elif 22.5 <= dP_value < 157.5:
        return round(dP_value / 45)+1


def make_EO_class(patches):
    EO_class = []
    for i in range (len(patches)):
        indices = []
        patch = patches[i]
        for n in range (2):
            for m in range (2):
                subpatch = patch[n:n+2, m:m+2]
                if mP(subpatch) < 15:
                    index = 0
                else:
                    index = dP(subpatch)
                indices.append(index)
        quainary = np.dot(indices, np.array([1, 5, 5**2, 5**3]).T)
        EO_class.append(quainary)
    return(EO_class)


def get_pairs_by_EO_class(EO_class, x, y, EO_class_index):
    pairs = []
    for i, EO_index in enumerate(EO_class):
        if EO_index == EO_class_index:
            pair = (x[i], y[i])
            pairs.append(pair)
    return pairs


def multi_variate_regression(X, Y):
    r = 1
    I = np.identity(Y.shape[0])
    return X@Y.T@np.linalg.inv(Y@Y.T +r*I)


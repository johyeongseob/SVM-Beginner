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
from SI_function import *
import glob
import pickle

image_paths = glob.glob("C:/Users/johs/image/SI/Kodak" + "/*.png")

H = []
L = []
y = []
x = []
for i in range(len(image_paths)):
    H.append(cv2.cvtColor(cv2.imread(image_paths[i]), cv2.COLOR_BGR2GRAY))
    L.append(SB(H, i))
    y.append(make_patch_y(L[i]))
    x.append(make_patch_x(H[i]))

y = np.array(y).reshape(-1, 3, 3)
x = np.array(x).reshape(-1, 2, 2)

EO_class = make_EO_class(y)
unique_EO_class = sorted(set(EO_class))

Mc = []
for EO_class_index in unique_EO_class:
    pairs_class = get_pairs_by_EO_class(EO_class, x, y, EO_class_index)
    Xc = []
    Yc = []

    for i, pair in enumerate(pairs_class):
        x_patch, y_patch = pair
        Xc.append(x_patch)
        Yc.append(y_patch)

    Xc = np.array(Xc).reshape(len(Xc), -1).T.astype(np.int32)
    Yc = np.array(Yc).reshape(len(Yc), -1).T.astype(np.int32)
    Map = multi_variate_regression(Xc, Yc)
    Mc.append((EO_class_index, Map))

with open('Mc.pickle', 'wb') as f:
    pickle.dump(Mc, f)


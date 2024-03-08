"""

Paper: Super-Interpolation With Edge-Orientation-Based Mapping Kernels for Low Complex 2x Upscaling / Jae-Seok Choi; Munchurl Kim / 09 December 2015

[ Up Scaling Phase ]

1. Input: LR input image L, pre-learned linear mappings M.
2. Initialization:
   Pad up LR image L with 1 pixels
   Generate 3x3 LR patches y.
3. For each LR input patch y, do
4.      For each 2x2 LR sub-patches in y, do
5.		    Apply 2 operators to compute its edge-direction index
6.	    End For
7.	    Compute EO class index C
8.	    Look-up corresponding linear Mapping Mc by its EO class index C.
9. 	    Apply obtained linear mappong Mc to the current LR patch y to reconstruct its HR version x
10. End For
11. Combine all reconstructed HR patches x.
12. Output: HR image H.

"""

import numpy as np
import cv2
from SI_function import *
import pickle
import os

with open('Mc.pickle', 'rb') as f:
    Mc = pickle.load(f)

output_folder = "C:/Users/johs/image/SI"

image_path = "C:/Users/johs/image/SI/girl.png"
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

resized_width = cv2.resize(image, (image.shape[1] // 2, image.shape[0]), interpolation=cv2.INTER_LINEAR)
resized_image = cv2.resize(resized_width, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

L = cv2.copyMakeBorder(resized_image, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)
width, height = L.shape[0]-2, L.shape[1]-2

y = make_patch_y(L)
EO_class = make_EO_class(y)

x = []
for i in range (len(y)):
    y_vector = np.array(y[i]).reshape(-1, 1).astype(np.int32)
    x_vector = (Mc[EO_class[i]][1]@y_vector).reshape(2,2).astype(np.uint8)
    x.append(x_vector)

HR = np.zeros((width*2, height*2), dtype=int)

for i in range(0, width*2, 2):
    for j in range(0, height*2, 2):
        small_matrix = x[i//2 * height + j//2]
        HR[i:i+2, j:j+2] = small_matrix

HR = cv2.convertScaleAbs(HR)

# print(HR)
# for i in range (HR.shape[0]):
#     for j in range (HR.shape[1]):
#         if HR[i][j] > 255:
#             print(HR[i][j])


output_file = os.path.join(output_folder, "HR.jpg")
resized_file = os.path.join(output_folder, "Resized.jpg")


# HR 이미지를 파일로 저장
cv2.imwrite(output_file, HR)
cv2.imwrite(resized_file, resized_image)


cv2.imshow("input Image", image)
cv2.imshow("resized Image", image)
cv2.imshow('HR_imgae', HR)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
K-means clustering

1. 임의의 image를 선택하여 배열 형태로 바꾼다.
2. 클러스터 점을 추가한다.

"""

from PIL import Image
import numpy as np
import math

image_path = "C:/johs/dog.jpg"

input_image = Image.open(image_path)

width = input_image.size[0]
height = input_image.size[1]

pixels = np.array(input_image).reshape(-1,3)

n = int(math.sqrt(pixels.size/3))

k = 3
centroids = pixels[np.random.choice(pixels.shape[0], size=k, replace=False)]

while True:
    clusters = []
    for pixel in pixels:
        distances = np.linalg.norm(pixel - centroids, axis=1) # 거리 계산
        cluster = np.argmin(distances) # 가장 가까운 값을 인덱스로 표현
        clusters.append(cluster)
    clusters = np.array(clusters)

    #clusters의 인덱스와 동일한 pixels의 인덱스에 해당하는 pixel들의 평균을 구한다.
    new_centroids = np.array([pixels[clusters==i].mean(axis=0) for i in range(k)])

    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids

output_pixels = centroids[clusters].astype(np.uint8)  # 클러스터 중심값으로 정수값픽셀 대체
output_image = Image.fromarray(output_pixels.reshape(width, height, 3))
output_image.show()

import numpy as np
from PIL import Image

# 이미지 로드
image_path = "C:/Users/johs/image/girl.png"
image = np.array(Image.open(image_path))
image = image.astype(np.int32)

# Bilateral 필터링 파라미터 설정
sigma_s = 30
sigma_r = 30

# 이미지의 크기 구하기
height, width, _ = image.shape

# 결과 이미지 초기화
filtered_image = np.zeros_like(image)

# 필터링 수행
for i in range(height):
    for j in range(width):
        pixel = image[i, j]

        # 현재 픽셀을 중심으로 윈도우 설정
        i_min = max(i - 2, 0)
        i_max = min(i + 3, height)
        j_min = max(j - 2, 0)
        j_max = min(j + 3, width)

        # 윈도우 내 픽셀 수집
        window = image[i_min:i_max, j_min:j_max]

        # 윈도우 내 픽셀과의 거리 계산
        spatial_dist = np.sqrt((i - np.arange(i_min, i_max)[:, None]) ** 2 + (j - np.arange(j_min, j_max)) ** 2)

        # 윈도우 내 픽셀과의 색상 차이 계산
        color_dist = np.sqrt(np.sum((window - pixel) ** 2, axis=2))

        # 가중치 계산
        weight = np.exp(-spatial_dist ** 2 / (2 * sigma_s ** 2) - color_dist ** 2 / (2 * sigma_r ** 2))

        # 정규화된 가중치 계산
        normalized_weight = weight / np.sum(weight)

        # 가중 평균 필터링 수행
        filtered_pixel = np.sum(window * np.expand_dims(normalized_weight, axis=2), axis=(0, 1))

        # 필터링된 픽셀 저장
        filtered_image[i, j] = filtered_pixel

# 결과 이미지 출력
result_image = Image.fromarray(filtered_image.astype(np.uint8))
result_image.show()

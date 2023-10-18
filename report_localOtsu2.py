#                       local Otsu thresholding
# <global Otsu thresholding>, <adaptive mean thresholding> 과 비교
# 위의 두 방법대비 local Otsu thresholding 효과가 잘 나타나는 영상을 찾아 테스트 (해상도는 640×640 이하)

# localOtsu 구현 후, globalOtsu와 adaptive mean과 비교

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
# import copy    


def localOtsu2(image, maxval, block_size_pixel = 10, C = 0):
    # 1. 이미지의 형태 파악.
    height, width = image.shape

    # 2. 결과를 최종 저장할 ZeroPadding 이미지 생성
    result = np.zeros([height, width], dtype=np.uint8)

    # 3. 이미지를 픽셀 단위로 순회하며, Local Otsu Thresholding을 적용
    for y in range(height):
        for x in range(width):
            # 현재 픽셀 위치에서 block을 정의 - 출발 및 마지막 고려 max, min 사용.
            block_y_start = max(0, y - block_size_pixel)
            block_x_start = max(0, x - block_size_pixel)
            
            block_y_end = min(height, y + block_size_pixel + 1)
            block_x_end = min(width, x + block_size_pixel + 1)

            # 해당 박스 내의 픽셀 값에 Local Otsu Thresholding을 적용
            block = image[block_y_start:block_y_end, block_x_start:block_x_end] - C
            thresholded_grid_pixel = cv.threshold(block, 0, maxval, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

            # Local Otsu Thresholding이 적용된 픽셀 값을 result에 대입.
            # pixel이 연속적으로 덮어씌워 이전 방법보다 연산량은 대폭 늘었지만, 훨씬 부드러움.
            result[y, x] = thresholded_grid_pixel[block_y_start - y + block_size_pixel, block_x_start - x + block_size_pixel]

    return result

img_gray = cv.imread("VisualComputing/japan.jpg", cv.IMREAD_GRAYSCALE)
# Adaptive_Mean
img_th1 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                    cv.THRESH_BINARY, 11, 2)
# Global_Otsu
th2v, img_th2 = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# Local_Otsu 
img_th3 = localOtsu2(img_gray, 255, 6 , 2)


print(img_gray.shape)

plt.subplot(2,2,1)
plt.imshow(img_gray, cmap='gray')
plt.title("Image")

plt.subplot(2,2,2)
plt.imshow(img_th3, cmap='gray')
plt.title("Local Otsu Thresholding2")

plt.subplot(2,2,3)
plt.imshow(img_th1, cmap="gray")
plt.title("Adaptive Mean Thresholding")

plt.subplot(2,2,4)
plt.imshow(img_th2, cmap="gray")
plt.title("Global Otsu Thresholding" + str(th2v))

plt.show()
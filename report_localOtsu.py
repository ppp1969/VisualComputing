#                       local Otsu thresholding
# <global Otsu thresholding>, <adaptive mean thresholding> 과 비교
# 위의 두 방법대비 local Otsu thresholding 효과가 잘 나타나는 영상을 찾아 테스트 (해상도는 640×640 이하)

# localOtsu 구현 후, globalOtsu와 adaptive mean과 비교

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
# import copy    

def localOtsu1(image, maxval, n, C = 0):
    # 1. 이미지의 형태 파악.
    height, width = image.shape

    # 2. 블럭을 입력받은 n에 따라 나눠준다.
    block_height = height // n
    block_width = width // n

    # 3. 결과를 최종 저장할 ZeroPadding 이미지 생성
    result = np.zeros([height, width])

    # 3-1. 반복문이 순회하며 격자가 이미지를 벗어나는 경우, 
    # 마지막 픽셀이 모자르면 Otsu 적용이 안되고, 검정색 픽셀로 계속 남아있음 
    # -> 처음에 result를 원본 이미지로 저장해놓고 싶다면 제로패딩이 아닌 원본을 복사해서 저장.
    # 예시 : result = copy.deepcopy(src) , 깊은 복사 사용해야함.

    
    # 4. 이미지를 격자 단위로 순회하며, Local Otsu Tresholding 적용
    for y in range(n+1):
        for x in range(n+1):
            # block 크기를 바탕으로 시작, 끝 좌표 지정
            y_start, y_end = y * block_height, (y + 1) * block_height
            x_start, x_end = x * block_width, (x + 1) * block_width

            # 해당 블럭을 Block 변수에 임시 저장.
            Block = image[y_start:y_end, x_start:x_end] - C

            # 블럭마다 Otsu 적용, [1]을 사용해 필요한 값만 받기.
            thresholded_grid = cv.threshold(Block, 0, maxval, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

            # Local Otsu Thresholding이 적용된 픽셀 값을 result에 대입.
            # 빈이미지에 블럭마다 합쳐짐.
            result[y_start:y_end, x_start:x_end] = thresholded_grid

    # (추가구현)
    # 3-2. 이미지의 h,w가 n으로 나눠떨어지지 않는 경우 
    # -> 직접 pixel 계산해서, 남는 영역 스레쉬홀딩을 적용하면 해결.

    unprocessed_height = height % n
    unprocessed_width = width % n

    # 3-2-1
    # 격자가 처리못한 가로줄 (남는 height에 대해서, block_width로 반복)
    for x in range(n):
        y_start, y_end = height - unprocessed_height, height
        x_start, x_end = x * block_width, (x + 1) * block_width

        Block = image[y_start:y_end, x_start:x_end] - C

        _, thresholded_grid = cv.threshold(Block, 0, maxval, cv.THRESH_BINARY | cv.THRESH_OTSU)

        result[y_start:y_end, x_start:x_end] = thresholded_grid

    # 3-2-2
    # 격자가 처리못한 세로줄 (남는 width에 대해서, block_height로 반복)
    for y in range(n):
        y_start, y_end = y * block_height, (y + 1) * block_height
        x_start, x_end = width - unprocessed_width, width

        Block = image[y_start:y_end, x_start:x_end] - C

        _, thresholded_grid = cv.threshold(Block, 0, maxval, cv.THRESH_BINARY | cv.THRESH_OTSU)

        result[y_start:y_end, x_start:x_end] = thresholded_grid    

    # 3-2-3 
    # 위 2개 경우 모두 block 크기로 진행시켰기에, 마지막 네모칸 남을 수 밖에 없음 -> 처리
    
    y_start, y_end = height - unprocessed_height, height
    x_start, x_end = width - unprocessed_width, width

    Block = image[y_start:y_end, x_start:x_end] - C

    _, thresholded_grid = cv.threshold(Block, 0, maxval, cv.THRESH_BINARY | cv.THRESH_OTSU)

    result[y_start:y_end, x_start:x_end] = thresholded_grid

    return result


img_gray = cv.imread("VisualComputing/japan.jpg", cv.IMREAD_GRAYSCALE)

# Adaptive_Mean
img_th1 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                    cv.THRESH_BINARY, 11, 2)
# Global_Otsu
th2v, img_th2 = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# Local_Otsu 
img_th3 = localOtsu1(img_gray, 255, 30 , 2)


print(img_gray.shape)

plt.subplot(2,2,1)
plt.imshow(img_gray, cmap='gray')
plt.title("Image")

plt.subplot(2,2,2)
plt.imshow(img_th3, cmap='gray')
plt.title("Local Otsu Thresholding1")

plt.subplot(2,2,3)
plt.imshow(img_th1, cmap="gray")
plt.title("Adaptive Mean Thresholding")

plt.subplot(2,2,4)
plt.imshow(img_th2, cmap="gray")
plt.title("Global Otsu Thresholding" + str(th2v))

plt.show()
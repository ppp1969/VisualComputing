import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

# png 파일도 rgv 값이 있을거임.
# 어찌됐건, 3채널짜리 이미지를 1채널로 읽어 분석할 것.
# 읽어올때부터 gray 스케일로 읽을 수 있음

# openCV에서 gray scale로 한 채널을 읽어왔지만!!!!!
# matplotlib에서는 한 채널의 값을 읽는게 기본값이 그레이스케일이 아님!! - viridius라 다르다.

# 그렇다면 우리가 기대하는 그레이 스케일을 출력하자면, matplotlib의 기본 컬러맵을
# gray로 맞춰줘야 함.

img_gray = cv.imread("VisualComputing/cameraman.png", cv.IMREAD_GRAYSCALE)

histo_gray = cv.calcHist([img_gray], [0], None, [256], [0,255])

plt.subplot(1,2,1)
# 컬러맵 그레이로 바꾸기!
plt.imshow(img_gray, cmap="gray")
plt.title("gray image")

plt.subplot(1,2,2)
plt.plot(histo_gray, color="black")
plt.xlim([0,255])

plt.show()
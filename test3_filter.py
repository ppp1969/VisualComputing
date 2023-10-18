import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_brg = cv.imread("VisualComputing/home.jpg")
img_rgb = cv.cvtColor(img_brg, cv.COLOR_BGR2RGB)

img_blur = cv.blur(img_rgb,(5,5))
img_gaussianblur = cv.GaussianBlur(img_rgb,(5,5),0)
# 시그마에 표준편차를 넣음. 0넣으면 디폴트를 호출.


# https://numpy.org/doc/stable/reference/generated/numpy.array.html
kernel = np.array(np.mat('0,-1,0;-1,5,-1;0,-1,0'))
# 채널 개수를 어떻게 가져가냐~~ -1넣으면 원본과 같음.
# 가우시안은 자기 자신에게 더 많은 가중치를 넣기에 덜 블러함.
img_sharp = cv.filter2D(img_rgb, -1, kernel)

plt.subplot(2,2,1), plt.imshow(img_rgb)
plt.title("input rgb image")
plt.subplot(2,2,2), plt.imshow(img_blur)
plt.title("blur image")
plt.subplot(2,2,3), plt.imshow(img_gaussianblur)
plt.title("gaussian blur image")
plt.subplot(2,2,4), plt.imshow(img_sharp)
plt.title("sharpen image")
plt.show()
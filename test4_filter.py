import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_brg = cv.imread("VisualComputing/home.jpg")
img_rgb = cv.cvtColor(img_brg, cv.COLOR_BGR2RGB)

img_blur = cv.blur(img_rgb,(3,3))

# https://numpy.org/doc/stable/reference/generated/numpy.array.html
# 얘는 총합이 0인 디테일 필터. 총합이 1이면 원본과 비슷.

# mat 함수는 배열로 만들어주는 함수.
kernel = np.array(np.mat('-1.0,-1.0,-1.0;-1.0,8.0,-1.0;-1.0,-1.0,-1.0'))
kernel_new = kernel / 9.0

# 원본에 디테일 필터를 취해주면 -> 샤프 해진다.(x)
# 원본에 디테일필터를 취한것 + 원본 -> 샤프필터

# 디테일 필터.
img_sharp = cv.filter2D(img_rgb, -1, kernel_new) * 3

# try and think why...
# img_sharp = cv.filter2D(img_rgb, -1, kernel / 3.0) ... same result okay.

# cv.subtract(img_rgb * 3, img_blur * 3) ?? why different?
# img_rgb * 3 - img_blur * 3 ?? why different?
# hint: data type

plt.subplot(2,2,1), plt.imshow(img_rgb)
plt.title("input rgb image")
plt.subplot(2,2,2), plt.imshow(img_blur)
plt.title("blur image")
# 원본 - 블러 = 디테일 필터 증명 / *3 는 값이 너무 안보여서 그냥 적용
plt.subplot(2,2,3), plt.imshow(cv.subtract(img_rgb, img_blur) * 3)
plt.title("gaussian blur image")
plt.subplot(2,2,4), plt.imshow(img_sharp)
plt.title("sharpen image")
plt.show()
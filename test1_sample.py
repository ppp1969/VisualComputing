import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# https://docs.opencv.org/4.8.0/da/d54/group__imgproc__transform.html
img_original = cv.imread("VisualComputing/image2.jpg")
img_original = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)

# 리사이즈 과정
size = (img_original.shape[:2])[::-1] # 컬러제외 y,x -> x,y로 읽어오기.
# 왜?? 리사이즈에 받는 투플 값은 width, height로 되어있기 때문에 거꾸로 읽음.

img_tmp = np.zeros((100,100,3), dtype=np.uint8)

rescale_size = tuple([int(x * 0.3) for x in size]) 
# 연산 자동형변환으로 float됨 -> int 형변환
# x for x in size에서 size가 2차원 배열인데 가능?
# => size[0], size[1] 이렇게 x가 나옴. 이때 넘파이는 1차원에 0.3 곱하면 전체가 곱해짐. 

#print (rescale_size)
img_resize = cv.resize(img_original, rescale_size, img_tmp, 0, 0, cv.INTER_LINEAR)

plt.subplot(1,2,1), plt.imshow(img_original)
plt.title("original image")
plt.subplot(1,2,2), plt.imshow(img_resize)
plt.title("resize image")
#plt.subplot(2,2,3), plt.imshow(img_gaussianblur)
#plt.title("gaussian blur image")
#plt.subplot(2,2,4), plt.imshow(img_sharp)
#plt.title("sharpen image")
plt.show() 
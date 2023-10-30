import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# openCV는 bgr순으로 넘파이 배열에 저장한다.
img = cv.imread("apple.jpg")

# 사이즈 확인
print(img.shape)

# 다운스케일 절반으로 줄음. 이때 홀수 값이라면 오차가 일어남! (반올림 해버림.)
img_lower_res = cv.pyrDown(img)
print(img_lower_res.shape)

# 다운스케일 직접 구현
# 1 가우시안 블러링 적용.
# 2 sub sampling, rescale 함수 활용

# 업스케일링 - 기존 이미지와 다를 수 있다. 홀수면!!!
img_higher_res1 = cv.pyrUp(img_lower_res)
print(img_higher_res1.shape)

# -> 직접 사이즈를 지정해서 업스케일링하자. 
# # 사이즈 : (width, height)이기에 y,x 관점을 바꿔야함.
img_higher_res2 = cv.pyrUp(img_lower_res,None, img.shape[0:2][::-1])
print(img_higher_res2.shape)

plt.subplot(2,2,1)
plt.imshow(img[...,::-1]) # 마지막 컬러 배열 값 bgr을 rgb로 읽어라.
plt.title("original")

plt.subplot(2,2,2)
plt.imshow(img_lower_res[...,::-1]) # 마지막 컬러 배열 값 bgr을 rgb로 읽어라.
plt.title("lower")

plt.subplot(2,2,3)
plt.imshow(img_higher_res2[...,::-1]) # 마지막 컬러 배열 값 bgr을 rgb로 읽어라.
plt.title("higher")

plt.subplot(2,2,4)
plt.imshow(cv.subtract(img[...,::-1], img_higher_res2[...,::-1])) # 마지막 컬러 배열 값 bgr을 rgb로 읽어라.
plt.title("img - higher = HighFrequency")

plt.show()
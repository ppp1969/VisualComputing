import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

img_gray = cv.imread("VisualComputing/sudoku.png", cv.IMREAD_GRAYSCALE)

th1v, img_th1 = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
th2v, img_th2 = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
img_th3 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY, 11, 2) # 뒤의 0은 너무 밝을때 일정 값을 전부 빼는것.

print(img_gray.shape)

plt.subplot(2,2,1)
plt.imshow(img_gray, cmap='gray')
plt.title("Image")

plt.subplot(2,2,2)
plt.imshow(img_th3, cmap='gray')
plt.title("Adaptive Image")

plt.subplot(2,2,3)
plt.imshow(img_th1, cmap="gray")
plt.title("Thresholding by value" + str(th1v))

plt.subplot(2,2,4)
plt.imshow(img_th2, cmap="gray")
plt.title("Otsu Thresholding" + str(th2v))

plt.show()
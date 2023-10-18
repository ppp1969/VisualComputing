import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# most image processing uses grayscaled image
# so, cv.IMREAD_GRAYSCALE is very useful for this
img_gray = cv.imread("j.png", cv.IMREAD_GRAYSCALE) 
img_hole_gray = cv.imread("j_holes.png", cv.IMREAD_GRAYSCALE) 
img_outn_gray = cv.imread("j_outside_noise.png", cv.IMREAD_GRAYSCALE) 

# https://numpy.org/doc/stable/reference/generated/numpy.ones.html
kernel = np.ones((5,5),np.uint8)
img_erosion = cv.erode(img_gray, kernel, iterations = 1)
img_dilation = cv.dilate(img_gray, kernel, iterations = 1)
img_opening = cv.morphologyEx(img_gray, cv.MORPH_OPEN, kernel)
img_closing = cv.morphologyEx(img_gray, cv.MORPH_CLOSE, kernel)
img_gradient = cv.morphologyEx(img_gray, cv.MORPH_GRADIENT, kernel)

img_opening_m = cv.erode(img_outn_gray, kernel, iterations = 1)
img_opening_m = cv.dilate(img_opening_m, kernel, iterations = 1)

img_closing_m = cv.dilate(img_hole_gray, kernel, iterations = 1)
img_closing_m = cv.erode(img_closing_m, kernel, iterations = 1)

img_gradient_m = img_dilation - img_erosion

def plot_img(index, img, title):
    plt.subplot(4,3,index), plt.imshow(img, cmap='gray'), plt.axis('off')
    plt.title(title)

plot_img(1, img_gray, "input image")
plot_img(2, img_erosion, "erosion")
plot_img(3, img_dilation, "dilation")
plot_img(4, img_outn_gray, "input image")
plot_img(5, img_opening, "opening")
plot_img(6, img_opening_m, "opening m")
plot_img(7, img_hole_gray, "input image")
plot_img(8, img_closing, "closing")
plot_img(9, img_closing_m, "closing m")
plot_img(10, img_gray, "input image")
plot_img(11, img_gradient, "gradient")
plot_img(12, img_gradient_m, "gradient m")

plt.show()
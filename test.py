import cv2 as cv

# 열어놓은 VC 폴더가 베이스기 때문에, 거기서 시작해야함.
img = cv.imread("VisualComputing/test_img.jpg")
cv.imshow("asd",img)

cv.waitKey(0)


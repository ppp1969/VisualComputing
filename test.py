import cv2 as cv

# 열어놓은 VC 폴더가 베이스기 때문에, 거기서 시작해야함.
img = cv.imread("VisualComputing/test_img.jpg") # 넘파이 배열로 읽혀짐.
cv.imshow("asd",img)

cv.waitKey(0)


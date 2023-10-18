import cv2 as cv
from matplotlib import pyplot as plt

# openCV는 bgr순으로 넘파이 배열에 저장한다.
img = cv.imread("VisualComputing/home.jpg")
# pyplot을 사용한다면, 역순으로 읽어야 rgb값이 정상적으로 매칭된다.

print(img.shape)

# [256]은 빈을 어느 정도의 개수로 가질건지 -> 빈이 얼나마 큰지, [0,256]은 값의 범위!
hist = cv.calcHist([img],[0],None,[256],[0,255])

# openCV는 BGR 순이니, RGB 각각채널 따로 해보자.
histo_r = cv.calcHist([img],[2],None, [256],[0,255])
histo_g = cv.calcHist([img],[1],None, [256],[0,255])
histo_b = cv.calcHist([img],[0],None, [256],[0,255])

# gray-scale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img_gray.shape)
histo_gray = cv.calcHist([img],[0],None, [256], [0, 255])

plt.subplot(1,2,1)
# (384, 512) - bgr 투플이 사라짐..!
plt.imshow(img[:,:,::-1])

plt.subplot(1,2,2)
plt.plot(histo_r, color = 'r')
plt.plot(histo_g, color = 'g')
plt.plot(histo_b, color = 'b')
plt.plot(histo_gray, color = 'b')

plt.title("rgb image")
plt.show()
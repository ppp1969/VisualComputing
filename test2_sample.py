import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
# Create a numpy array, pretending it is our image
img = np.array([[3, 106, 107, 40, 148, 112, 254, 151],
                [62, 173, 91, 93, 33, 111, 139, 25],
                [99, 137, 80, 231, 101, 204, 74, 219],
                [240, 173, 85, 14, 40, 230, 160, 152],
                [230, 200, 177, 149, 173, 239, 103, 74],
                [19, 50, 209, 82, 241, 103, 3, 87],
                [252, 191, 55, 154, 171, 107, 6, 123],
                [7, 101, 168, 85, 115, 103, 32, 11]],
                dtype=np.uint8)
# Resize the width and height in half

# 2x2씩 묶임.

# Area 보간법이 어떻게 축소되는 지 보여줌.
resized = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), 0, 0,
                     interpolation=cv.INTER_AREA)
print(resized)
# Result:
# [[ 86  83 101 142]
# [162 103 144 151]
# [125 154 189  67]
# [138 116 124  43]]


img_original = cv.imread("VisualComputing/shirts.png")
print (cv.version)
img_original = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
img_tmp = np.zeros((100,100,3), dtype=np.uint8)
size = (img_original.shape[:2])[::-1]
rescale_size = tuple([int(x * 0.3) for x in size])
#print (rescale_size)

# high frequency 이미지의 경우 Bad Sampling이 생기기 쉽다. 기존의 패턴을 잃어버림.
# Linear은 양옆에 하나씩 가져와서 그걸 이어줌 -> 스트라이프 패턴은 망한다!!!
# Area는 양옆의 샘플 100개씩 가져와서 그 평균을 사용.
# 대부분은 선형을 쓰지만, 축소 및 하이프리퀀시 이미지는 Area를 사용해라~
# 이미지를 3배 키운다고 가정했을 때, Area는 Nearest와 같다.(빵꾸가 있으니까)
img_resize = cv.resize(img_original, rescale_size, img_tmp, 0, 0, cv.INTER_LINEAR)
img_resize2 = cv.resize(img_original, rescale_size, img_tmp, 0, 0, cv.INTER_AREA)
img_resize3 = cv.resize(img_original, rescale_size, img_tmp, 0, 0, cv.INTER_NEAREST)

# 내 생각엔 확대 시에 AREA의 읽어오는 픽셀에선 공백이 많을거임..
# 그럼 가장 가까운 픽셀이 가중치가 높을 수 밖에 업고, window 평균으로 읽어오니
# 0으로 엄청 채워진 업스케일링된 이미지에서 보간법을 사용하면 Near과 같은 효과.

plt.subplot(2,2,1), plt.imshow(img_original)
plt.title("original image")
plt.subplot(2,2,2), plt.imshow(img_resize2)
plt.title("resize image")
plt.subplot(2,2,3), plt.imshow(img_resize)
plt.title("resize image (area)")
plt.subplot(2,2,4), plt.imshow(img_resize3)
plt.title("resize image (nearest)")
plt.show()
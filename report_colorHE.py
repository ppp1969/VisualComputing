import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('tree.jpg')

print(img.shape)

assert img is not None, "file could not be read, check with os.path.exists()"

def make_cdf(image = img):
    hist, bins = np.histogram(image.flatten(), 256, [0,255]) 
    cdf = hist.cumsum() # cdf of a pdf
    cdf_normalized = cdf / cdf.max() # cdf 0~1 비율로 바꾸기.
    cdf_display =  cdf_normalized * float(hist.max()) # 히스토그램에 맞게 디스플레이.
    
    return cdf, cdf_display

# 1. hsv 컬러 이용
#       hsv에서 v채널을 추출하여, HE적용.
#       V(Value) 명도 : 색이 진한 정도
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# h, s, v로 컬러 영상을 분리 합니다. 
h, s, v = cv.split(hsv)

cdf_img, cdf_display_img = make_cdf(v) # 기본 이미지 cdf 추출

#----Global
# v값에 대해서 global HE 적용!
HE_global_v = cv.equalizeHist(v)
cdf_HE_hsv_Dst, cdf_display_HE_hsv_Dst = make_cdf(HE_global_v) # global HE 이미지 cdf 추출


# h,s,equalizedV를 합쳐서, 새로운 hsv 이미지 만든 후 BGR로 변경.
HE_hsv = cv.merge([h,s,HE_global_v])
HE_hsv_Dst = cv.cvtColor(HE_hsv, cv.COLOR_HSV2BGR)
#----


h, s, v = cv.split(hsv)
#----Local
# v값에 대해서 adaptive HE 적용!
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
HE_local_v = clahe.apply(v)

cdf_AHE_hsv_Dst, cdf_display_AHE_hsv_Dst = make_cdf(HE_local_v) # local HE 이미지 cdf 추출

# h,s,equalizedV를 합쳐서 새로운 hsv 이미지 만듦.
AHE_hsv = cv.merge([h,s,HE_local_v])

# 마지막으로 hsv2를 다시 BGR 형태로 변경합니다.
AHE_hsv_Dst = cv.cvtColor(AHE_hsv, cv.COLOR_HSV2BGR)
#----




plt.subplot(2,3,1)
plt.imshow(img[...,::-1])
plt.title("Raw Image")


plt.subplot(2,3,4)
plt.plot(cdf_display_img, color = 'b')
plt.hist(v.flatten(), 255, [0,255], color='r') # 1000은 bin의 개수 0~255는 값 범위
plt.xlim([0,255])
plt.legend(('cdf_v', 'hist_v'), loc = 'upper left')
plt.title("Raw Image")

plt.subplot(2,3,2)
plt.imshow(HE_hsv_Dst[...,::-1])
plt.title("Global HE")

plt.subplot(2,3,5)
plt.plot(cdf_display_HE_hsv_Dst, color = 'b')
plt.hist(HE_global_v.flatten(), 255, [0,255], color='r') # 1000은 bin의 개수 0~255는 값 범위
plt.xlim([0,255])
plt.legend(('cdf_v', 'hist_v'), loc = 'upper left')
plt.title("Global HE")

plt.subplot(2,3,3)
plt.imshow(AHE_hsv_Dst[...,::-1])
plt.title("Adaptive HE")

plt.subplot(2,3,6)
plt.plot(cdf_display_AHE_hsv_Dst, color = 'b')
plt.hist(HE_local_v.flatten(), 255, [0,255], color='r') # 1000은 bin의 개수 0~255는 값 범위
plt.xlim([0,255])
plt.legend(('cdf_v', 'hist_v'), loc = 'upper left')
plt.title("Adaptive HE")

plt.show()
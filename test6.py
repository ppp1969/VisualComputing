import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt


img = cv.imread('VisualComputing/wiki.jpg', cv.IMREAD_GRAYSCALE)

assert img is not None, "file could not be read, check with os.path.exists()"

def make_cdf(image = img):
    hist, bins = np.histogram(image.flatten(), 256, [0,255]) 
    cdf = hist.cumsum() # cdf of a pdf
    cdf_normalized = cdf / cdf.max() # cdf 0~1 비율로 바꾸기.
    cdf_display =  cdf_normalized * float(hist.max()) # 히스토그램에 맞게 디스플레이.
    
    return cdf, cdf_display






cdf, cdf_display = make_cdf(img)

cdf_m = np.ma.masked_equal(cdf, 0) # cdf가 0인 경우, 제외 - 1부터 시작!
# 이걸 왜하냐면, cdf가 0인 경우도 모두 포함하기 때문에


# HE 핵심
# filter 역할 - 이 자체가 함수가 된다. 0부터 시작하는 것이 아니라. 1인값부터 시작
cdf_new = ((cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min())) * 255
cdf_ui8 = cdf_new.astype(np.uint8)

img_he = cdf_ui8[img] # 여기서 대입을 통해 HE 진행
img_he2 = cv.equalizeHist(img)
cdf_he, cdf_display_he = make_cdf(img_he2) # 성능이 빠를거임.

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')

plt.subplot(2,2,2)
plt.plot(cdf_display, color = 'b')
plt.hist(img.flatten(), 255, [0,255], color='r') # 1000은 bin의 개수 0~255는 값 범위
plt.xlim([0,255])
plt.legend(('cdf', 'hist'), loc = 'upper left')

plt.subplot(2,2,3)
plt.imshow(img_he, cmap='gray')

plt.subplot(2,2,4)
plt.plot(cdf_display_he, color = 'b')
plt.hist(img_he.flatten(), 255, [0,255], color='r') # 1000은 bin의 개수 0~255는 값 범위
plt.xlim([0,255])
plt.legend(('cdf', 'hist'), loc = 'upper left')





plt.show()

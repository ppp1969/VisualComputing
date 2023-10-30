import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def generate_gaussian_pyramid(img, levels): # 레벨 총개수. 원본 : 0부터 시작
    GP = [img] # 0 level.
    
    for i in range(1,levels):
        img_lower = cv.pyrDown(GP[-1])
        GP.append(img_lower)
     
    return GP

def generate_laplacian_pyramid(GP):
    # 라플라시안 피라미드는 각 레벨에서 하이프리퀀시 추출한 피라미드임.

    # 현재레벨 라플라시안 = [현재 레벨] - [1증가된 레벨]
    levels = len(GP)
    LP = []

    for i in range(levels - 1, 0, -1): # 0레벨에서 시작한다고 생각x
                                       # 마지막 레벨에서 시작해서 1레벨까지.
        img_higher = cv.pyrUp(GP[i], None, GP[i-1].shape[:2][::-1]) # w,h 생각.
        img_lap = cv.subtract(GP[i-1], img_higher) * 10
        LP.append(img_lap)
    
    LP.reverse()
    return LP




def generate_pyramid_composition_image(Pimgs):
    # Pimg = 가우시안 피라미드 이미지
    
    levels = len(Pimgs)
    rows, cols = Pimgs[0].shape[:2]

    # 사이즈가 만약 안맞더라도 반올림해서 더함! 이러면 0 or 1 더해져서 오류 잡힘.
    composite_image = np.zeros((rows, cols + int(cols / 2 + 0.5), 3), dtype=Pimgs[0].dtype)
    
    # 원본 스근하게 채워주고.
    composite_image[:rows, :cols, :] = Pimgs[0]

    # stack all downsampled images in a column to the right of the original
    i_row = 0
    for p in Pimgs[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row : i_row + n_rows, cols : cols + n_cols] = p
        i_row += n_rows    # 행은 계속 누적됨, 열은 매번 원본이미지에서 시작!

    return composite_image



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






GP = generate_gaussian_pyramid(img, 6) # 6레벨 피라미드 이미지 배열 생성.
LP = generate_laplacian_pyramid(GP)

img_GP_pistack = generate_pyramid_composition_image(GP)
img_LP_pistack = generate_pyramid_composition_image(LP)

cv.imshow("GPI",img_GP_pistack)
cv.imshow("LPI",img_LP_pistack)



















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
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def plot_img(rows_cols, index, img, title):
    plt.subplot(rows_cols[0], rows_cols[1], index)
    plt.imshow(img[...,::-1])
    plt.axis('off'), plt.title(title)

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
        img_lap = cv.subtract(GP[i-1], img_higher)
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


def stitch_LhalgAndRhalf(P_left, P_right):
    P_stitch = []
    for la, lb in zip(P_left, P_right):
        cols = la.shape[1]
        lpimg_stitch = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
        P_stitch.append(lpimg_stitch)
    return P_stitch






img_apple = cv.imread("apple.jpg")
img_orange = cv.imread("orange.jpg")

rows, cols = img_apple.shape[:2]
# 넘파이에서의 접근 방식은 y,x
img_halfhalf = np.hstack((img_apple[:,0:cols//2], img_orange[:,cols//2:])) 
    # 전처리 - 이미지 사이즈 맞추는 것 중요함. 사이즈를 맞추는 과정도 자동화 필요!
    # img_orange = cv.resize(img_orange, img_apple.shape[:2][::-1])

GP_apple = generate_gaussian_pyramid(img_apple, 6)
GP_orange = generate_gaussian_pyramid(img_orange, 6)
LP_apple = generate_laplacian_pyramid(GP_apple)
LP_orange = generate_laplacian_pyramid(GP_orange)

GP_halfNhalf = stitch_LhalgAndRhalf(GP_apple, GP_orange)
LP_halfNhalf = stitch_LhalgAndRhalf(LP_apple, LP_orange)




# 합친 마지막 레벨의 가우시안 피라미드와 LP 합친 친구들을 이용한다.


# 6번만하면 0레벨 도달 가능.

blending4 = cv.add(cv.pyrUp(GP_halfNhalf[-1]), LP_halfNhalf[4])
blending3 = cv.add(cv.pyrUp(blending4, None, GP_halfNhalf[3].shape[0:2][::-1]), LP_halfNhalf[3])
blending2 = cv.add(cv.pyrUp(blending3, None, GP_halfNhalf[2].shape[0:2][::-1]), LP_halfNhalf[2])
blending1 = cv.add(cv.pyrUp(blending2, None, GP_halfNhalf[1].shape[0:2][::-1]), LP_halfNhalf[1])
blending0 = cv.add(cv.pyrUp(blending1, None, GP_halfNhalf[0].shape[0:2][::-1]), LP_halfNhalf[0])

plt_rc = (2,3)


# 뿌연 결과에 Global HE 적용 후 살펴보기.
result_hsv = cv.cvtColor(blending0, cv.COLOR_BGR2HSV)
 
h, s, v = cv.split(result_hsv)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
HE_local_v = clahe.apply(v)

HE_hsv = cv.merge([h,s,HE_local_v])
HE_result = cv.cvtColor(HE_hsv, cv.COLOR_HSV2BGR)


plot_img(plt_rc, 1, HE_result, "HE result")
plot_img(plt_rc, 2, blending4, "level 4")
plot_img(plt_rc, 3, blending3, "level 3")
plot_img(plt_rc, 4, blending2, "level 2")
plot_img(plt_rc, 5, blending1, "level 1")
plot_img(plt_rc, 6, blending0, "level 0")


plt.show()

# 마지막 결과가 팅트가 된듯한 결과 : 뿌얘~~~ 교수님이 말한건 대비가 좀 작아진거 같다 ㅋ 
# -> CLAHE 적용 하면 되겠지. -> 정답.
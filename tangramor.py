import cv2 as cv
import numpy as np
# 加载两张图片
img1 = cv.imread('../meixi.png')
img2 = cv.imread('../cvb.png')
cv.imshow('cv', img2)
# 把logo放在左上角
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]
cv.imshow('roi', roi)

# 创建logo的掩码，并同时创建其相反掩码
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
cv.imshow('2gray', img2gray)
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
cv.imshow('mask', mask)
mask_inv = cv.bitwise_not(mask)
cv.imshow('mask_inv', mask_inv)
# logo区域涂黑  掩码白色区域保留，黑色剔除
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
cv.imshow('img1_bg', img1_bg)
# 提取logo区域
img2_fg = cv.bitwise_and(img2, img2, mask=mask)
cv.imshow('img2_fg', img2_fg)
# 修改主图像
dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst
cv.imshow('res', img1)
cv.waitKey(0)
cv.destroyAllWindows()


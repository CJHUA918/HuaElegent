import cv2 as cv
import numpy as np


def read_demo():
    img1 = cv.imread("./0014.jpg")  # filename, flags
    cv.imshow("image", img1)  # winname, mat
    cv.waitKey(0)
    cv.destroyAllWindows()


def color_space_demo():  # 色彩空间的转换
    image = cv.imread("./0014.jpg")  # BGR 0 ~ 255
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)  # H 0 ~ 180
    cv.imshow("gray", gray)
    cv.imshow("hsv", hsv)
    cv.waitKey(0)
    cv.destroyAllWindows()


def mat_demo():  # 对象的创建与赋值
    # image = cv.imread("./0014.jpg", cv.IMREAD_GRAYSCALE)
    image = cv.imread("./0014.jpg")
    print(image)
    print(image.shape)  # H W C
    roi = image[100:200, 100:200, :]
    blank = np.zeros_like(image)
    blank[60:200, 60:200, :] = image[60:200, 60:200, :]
    cv.imshow("blank", blank)
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # h, w, c = image.shape
    # roi = image[60:200, 60:200, :]
    # blank = np.zeros((h, w, c), dtype=np.uint8)
    # blank = np.copy(image)
    # cv.imshow("blank", blank)
    # cv.imshow("roi", roi)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def pixel_demo():  # 图像像素的读写操作
    image = cv.imread("./0014.jpg")
    cv.imshow("input", image)
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            b, g, r = image[row, col]
            image[row, col] = (255-b, 255-g, 255-r)
    cv.imshow("result", image)
    #  cv.imwrite("./result.jpg", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def math_demo():  # 图像像素的算术操作 图像大小和通道数需要一致
    image = cv.imread("./0014.jpg")
    cv.imshow("input", image)
    h, w, c = image.shape
    blank = np.zeros_like(image)
    blank2 = np.zeros_like(image)
    blank[:, :] = (50, 50, 50)
    blank2[:, :] = (2, 2, 2)
    cv.imshow("blank", blank)
    result = cv.add(image, blank)
    # result = cv.subtract(image, blank)
    result2 = cv.multiply(image, blank2)
    # result2 = cv.divide(image, blank2)
    cv.imshow("result", result)
    cv.imshow("result2", result2)
    cv.waitKey(0)
    cv.destroyAllWindows()


def nothing(x):
    print(x)


def adjust_lightness_demo():  # 滚动条调整图像亮度
    image = cv.imread("./0014.jpg")
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("lightness", "input", 0, 100, nothing)
    cv.imshow("input", image)
    blank = np.zeros_like(image)
    while True:
        pos = cv.getTrackbarPos("lightness", "input")
        blank[:, :] = (pos, pos, pos)
        result = cv.add(image, blank)
        cv.imshow("result", result)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()


def adjust_contrast_demo():  # 参数传递与调整亮度与对比度
    image = cv.imread("./0014.jpg")
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("lightness", "input", 0, 100, nothing)
    cv.createTrackbar("contrast", "input", 100, 200, nothing)
    cv.imshow("input", image)
    blank = np.zeros_like(image)
    while True:
        light = cv.getTrackbarPos("lightness", "input")
        contrast = cv.getTrackbarPos("contrast", "input")/100
        blank[:, :, :] = (light, light, light)
        print("light:", light, "contrast:", contrast)
        result = cv.addWeighted(image, contrast, blank, 0.5, light)
        cv.imshow("result", result)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()


def key_demo():  # 键盘响应操作
    image = cv.imread("./0014.jpg")
    cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
    cv.imshow("input", image)
    while True:
        c = cv.waitKey(1)
        if c == 49:  # 1
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            cv.imshow("result", gray)
        if c == 50:  # 2
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            cv.imshow("result", hsv)
        if c == 51:  # 3
            invert = cv.bitwise_not(image)
            cv.imshow("result", invert)
        if c == 27:
            break
    cv.destroyAllWindows()


if __name__ == '__main__':
    key_demo()

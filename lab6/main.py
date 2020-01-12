import cv2
import matplotlib.pyplot as plt
import numpy as np


# 1.读入图片，转化为HSV格式，提取出蓝色通道，转化为灰度图
# 2.平滑去噪
# 3.边缘检测
# 4.开闭计算，形成矩形区域
# 5.检测所有可能的轮廓，找到最大的区域即为车牌所在区域，用框标出

def read_photo_to_blue(img_array, lest_blue):
    """图片先转化为HSV空间，然后输出蓝色通道的灰度图"""
    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)  # 转化为hsv空间
    blue_low = np.array([100, 43, 46])  # 蓝色较低的值，由统计数据得出
    blue_high = np.array([124, 255, 255])  # 蓝色较高的值，由统计数据得出
    mask = cv2.inRange(hsv_img, blue_low, blue_high)  # 设阈值，阈值外的为0
    blue_img = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)  # 执行阈值的与运算，提取出蓝色部分的图
    blue_gray = cv2.cvtColor(blue_img, cv2.COLOR_BGR2GRAY)  # 将蓝色部分的图转化为灰度图
    # blue_gray = cv2.equalizeHist(blue_gray)  # 直方图均衡化
    for i in range(len(blue_gray[:, 0])):
        for j in range(len(blue_gray[0, :])):
            if blue_gray[i][j] < lest_blue:
                blue_gray[i][j] = 0
    plt.imshow(blue_gray, cmap='gray')
    plt.title('blue gray')
    plt.axis('off')
    plt.show()
    return blue_gray


def remove_noise(img_array):
    """对图片进行去噪（平滑）处理"""
    gauss_smooth_img = cv2.GaussianBlur(img_array, (9, 9), 1)  # 高斯平滑
    plt.imshow(gauss_smooth_img, cmap='gray')
    plt.title('after smooth')
    plt.axis('off')
    plt.show()
    return gauss_smooth_img


def edge_detect(img_array):
    """图片的边缘检测"""
    low = 10
    high = 10
    canny_img = cv2.Canny(img_array, low, high)
    plt.imshow(canny_img, cmap='gray')
    plt.title('low:' + str(low) + ',high:' + str(high))
    plt.title('edge detect')
    plt.axis('off')
    plt.show()
    return canny_img


def shut(img_array, step):
    """先执行闭合操作，将车牌填充；然后执行开启操作，将不是车牌的地方断开"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (step, step))  # 闭合操作的参数设置
    close_img = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)  # 执行闭合操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # 开启
    open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)
    plt.subplot(121)
    plt.imshow(close_img, cmap='gray')
    plt.title('after close')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(open_img, cmap='gray')
    plt.title('after open')
    plt.axis('off')
    plt.show()
    return open_img


def find_licence_plate(img_array):
    """寻找车牌的算法,检测轮廓，找出面积最大的轮廓"""
    # 寻找边框
    max_area = 0
    max_edge_num = 0
    contours, hierarchy = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            max_area = area
            max_edge_num = i
    rect = cv2.minAreaRect(contours[max_edge_num])
    box = np.int32(cv2.boxPoints(rect))
    return box


if __name__ == '__main__':
    file = '4.jpg'
    bgr_img = cv2.imread(file)  # 读入图片,bgr通道
    # door = 150  # 蓝色的最小值,图1,2,3
    door = 100  # 图4
    # door = 200  # 图5
    blue_array = read_photo_to_blue(bgr_img, door)  # 分离出蓝色通道
    smooth_img = remove_noise(blue_array)  # 去噪
    edge_img = edge_detect(smooth_img)  # 边缘检测
    steps = 18  # 图1,2,3,4
    # steps = 6  # 5
    morph_img = shut(edge_img, steps)  # 开闭操作
    edge_box = find_licence_plate(morph_img)
    img = cv2.drawContours(bgr_img, [edge_box], -1, (0, 255, 0), 1)
    cv2.imshow("img", bgr_img)
    cv2.waitKey(0)
    cv2.imwrite('output_' + file, img)

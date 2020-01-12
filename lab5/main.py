import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MAX_NUM = 256

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来plt正常显示中文标签


def histogram(img_array):
    """直方图均衡化"""
    # 1. 统计各灰度值的像素数目
    p = [0 for i in range(MAX_NUM)]  # 初始化各个灰度值的个数为0
    len_x = len(img_array[:, 0])
    len_y = len(img_array[0, :])
    for i in range(len_x):
        for j in range(len_y):
            p[img_array[i][j]] += 1  # 计数
    # 2.计算各灰度值出现的概率
    p = [x / (len_x * len_y) for x in p]
    # 3.计算累计分布函数
    c = [0 for i in range(MAX_NUM)]
    c[0] = p[0]
    for i in range(1, MAX_NUM):
        c[i] = c[i - 1] + p[i]
    # 4.计算映射后的灰度图
    after_hist = np.zeros((len_x, len_y), dtype='uint8')
    for i in range(len_x):
        for j in range(len_y):
            after_hist[i][j] = (MAX_NUM - 1) * c[img_array[i][j]] + 0.5
    # 5.统计变换后各灰度值出现的次数
    p_after = [0 for i in range(MAX_NUM)]
    for i in range(len_x):
        for j in range(len_y):
            p_after[after_hist[i][j]] += 1
    # 6.统计变换后各点出现的频率
    p_after = [x / (len_x * len_y) for x in p_after]
    x = [i for i in range(MAX_NUM)]

    # 绘制均衡化前的直方图
    plt.subplot(1, 2, 1)
    plt.hist(img_array.reshape(1, -1)[0], x, histtype='bar', color='g')
    plt.title('before')
    plt.xlabel('gray')
    plt.ylabel('f')
    # 绘制均衡化后的直方图
    plt.subplot(1, 2, 2)
    plt.hist(after_hist.reshape(1, -1)[0], x, histtype='bar', color='g')
    plt.title('after')
    plt.xlabel('gray')
    plt.ylabel('f')
    plt.show()
    # 绘制处理前的照片
    plt.subplot(1, 2, 1)
    plt.imshow(Image.fromarray(img_array), cmap='gray')
    plt.title('before')
    # 绘制处理后的照片
    plt.subplot(1, 2, 2)
    plt.imshow(Image.fromarray(after_hist), cmap='gray')
    plt.title('after')
    plt.show()


def dft(img_array):
    """傅里叶变换，显示幅度图、相位图"""
    dft_array = cv2.dft(np.float32(img_array), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 将图片的低频部分移动到中心位置
    dft_shift = np.fft.fftshift(dft_array)
    # cv2.magnitude()计算矩阵的加权平方根，将实部和虚部投影到空间域
    magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    # 计算相角
    len_x = len(img_array[:, 0])
    len_y = len(img_array[0, :])
    phase = []
    for i in range(len_x):
        for j in range(len_y):
            phase.append(np.complex(dft_shift[i, j, 0], dft_shift[i, j, 1]))
    phase = (180.0 * np.angle(phase) / np.pi + 2 * np.pi) % 360 / 360  # 相位信息,归一化
    phase = np.array(phase).reshape(len_x, len_y)  # 重构矩阵

    # 显示幅度图
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('dft magnitude')
    plt.axis('off')
    plt.show()

    # 显示相位图
    plt.imshow(phase, cmap='gray')
    plt.title('phase')
    plt.axis('off')
    plt.show()


def high_pass(img_array):
    """理想高通滤波"""
    dft_array = cv2.dft(np.float32(img_array), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 将图片的低频部分移动到中心位置
    dft_shift = np.fft.fftshift(dft_array)
    # 构造掩膜（中心为0，其余为1）
    len_x = len(img_array[:, 0])
    len_y = len(img_array[0, :])
    mask = np.ones((len_x, len_y, 2))
    for i in range(int(len_x / 2 - 10), int(len_x / 2 + 10)):
        for j in range(int(len_y / 2 - 10), int(len_y / 2 + 10)):
            mask[i][j][0] = 0  # 中心置0
            mask[i][j][1] = 0
    # 添加掩膜
    mask_img = np.multiply(mask, dft_shift)
    # 高低频恢复正常排布
    de_shift = np.fft.ifftshift(mask_img)
    # 傅里叶反变换
    de_dft = cv2.idft(de_shift)
    # 将图像转为空间域
    dft_img = cv2.magnitude(de_dft[:, :, 0], de_dft[:, :, 1])
    # 原图
    plt.subplot(121)
    plt.imshow(img_array, cmap='gray')
    plt.title('before')
    plt.axis('off')
    # 理想高通滤波后的图
    plt.subplot(122)
    plt.imshow(dft_img, cmap='gray')
    plt.title('after')
    plt.axis('off')
    plt.show()


def gauss_low_pass(img_array, sigma):
    """高斯低通滤波"""
    dft_array = cv2.dft(np.float32(img_array), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 将图片的低频部分移动到中心位置
    dft_shift = np.fft.fftshift(dft_array)
    # 构造掩膜（中心为0，其余为1）
    len_x = len(img_array[:, 0])
    len_y = len(img_array[0, :])
    # 添加高斯低通滤波
    for i in range(len_x):
        for j in range(len_y):
            d = np.exp(-((i - len_x / 2) ** 2 + (j - len_y / 2) ** 2) / (2 * sigma ** 2))
            dft_shift[i][j][0] *= d
            dft_shift[i][j][1] *= d
    # 高低频归位
    de_shift = np.fft.ifftshift(dft_shift)
    # 反傅里叶变换
    de_dft = cv2.idft(de_shift)
    # 将图像映射至空间域
    dft_img = cv2.magnitude(de_dft[:, :, 0], de_dft[:, :, 1])
    plt.subplot(121)
    plt.imshow(img_array, cmap='gray')
    plt.title('before')
    plt.axis('off')
    # 理想高通滤波后的图
    plt.subplot(122)
    plt.imshow(dft_img, cmap='gray')
    plt.title('after,sigma=' + str(sigma))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    img_array = np.array(Image.open('lena.jpg').convert('L'))  # 灰度图，灰度取值为0-255
    # histogram(img_array)  # 直方图均衡化
    # dft(img_array)  # 傅里叶变换
    # high_pass(img_array)  # 理想高通滤波
    sigma = 70  # sigma越小，图像越模糊
    gauss_low_pass(img_array, sigma)  # 高斯低通滤波

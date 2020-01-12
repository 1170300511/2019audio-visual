from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
import cv2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来plt正常显示中文标签


def add_gauss_noise(img_array, mean, cov):
    """添加高斯噪声"""
    # input: img_array 二维数组      待添加噪声的图像
    #         mean     一维数组   高斯噪声的均值（在这里为1维）
    #         cov      二维矩阵   高斯噪声的协方差（这里的高斯噪声为一维的，所以这里也是方差）
    # output: gauss_array  二维数组  经过高斯噪声处理后的数据

    gauss_noise = np.random.multivariate_normal(mean, cov, img_array.shape)[:, :, 0]  # 高斯噪声
    gauss_array = img_array + gauss_noise  # 添加噪声
    return gauss_array


def add_impulse_noise(img_array, SNR):
    """添加椒盐噪声"""
    # input: img_array  二维数组   待处理的图片
    #        SNR        double 信噪比
    # output: impulse_array  二维数组  加了椒盐噪声处理后的图片

    row = img_array.shape[0]  # 行数
    line = img_array.shape[1]  # 列数
    noise_size = int(img_array.shape[0] * img_array.shape[1] * (1 - SNR))  # 椒盐噪点的个数
    impulse_array = img_array
    # 添加椒盐噪声
    for i in range(noise_size):
        i = np.random.randint(0, row)  # 行标号
        j = np.random.randint(0, line)  # 列标号
        if i % 2 == 0:
            impulse_array[i][j] = 0
        else:
            impulse_array[i][j] = 255
    return impulse_array


def add_noise():
    """第一部分：添加高斯噪声和椒盐噪声,并将加噪声的图打印出来"""
    img = Image.open('lina.jpg').convert('L')  # 读入图片，图片的尺寸为128x128
    img_array = np.array(img)  # 转化为array
    cov = [[500]]  # 方差
    mean = [0]  # 均值
    gauss_array = add_gauss_noise(img_array, mean, cov)  # 添加高斯噪声
    SNR = 0.8  # 信噪比
    impulse_array = add_impulse_noise(img_array, SNR)  # 添加椒盐噪声

    plt.figure()
    # 原图
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('原图')
    plt.axis('off')
    # 加高斯噪声的图
    plt.subplot(1, 3, 2)
    plt.imshow(Image.fromarray(gauss_array))
    plt.title('加高斯噪声,mean=' + str(mean) + ',cov=' + str(cov))
    plt.axis('off')
    # 加椒盐噪声的图片
    plt.subplot(1, 3, 3)
    plt.imshow(Image.fromarray(impulse_array), cmap='gray')
    plt.title('加椒盐噪声,SNR=' + str(SNR))
    plt.axis('off')

    plt.show()  # 显示


def mid_filter(img_array):
    """中值滤波"""
    mid_array = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 中值滤波矩阵
    rel_array = np.zeros((img_array.shape[0] - 2, img_array.shape[1] - 2))
    for i in range(1, img_array.shape[0] - 1):
        for j in range(1, img_array.shape[1] - 1):
            x_array = img_array[i - 1:i + 2, j - 1:j + 2]
            # 卷积，寻找中值
            rel_array[i - 1][j - 1] = sorted(np.multiply(mid_array, x_array).reshape(1, -1)[0])[5]
    return rel_array


def con(kernel, array):
    """卷积运算"""
    rel_array = np.zeros((array.shape[0] - 2, array.shape[1] - 2))
    for i in range(1, array.shape[0] - 1):
        for j in range(1, array.shape[1] - 1):
            rel_array[i - 1, j - 1] = sum(np.multiply(kernel, array[i - 1:i + 2, j - 1:j + 2]).reshape(1, -1)[0])
    return rel_array


def aver_filter(img_array):
    """均值滤波"""

    aver_array = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
    return con(aver_array, img_array)


def smooth():
    """第二部分：图像平滑"""
    img = Image.open('lina.jpg').convert('L')  # 读入图片，图片的尺寸为128x128
    img_array = np.array(img)  # 转化为array
    cov = [[300]]  # 方差
    mean = [0]  # 均值
    gauss_array = add_gauss_noise(img_array, mean, cov)  # 添加高斯噪声
    mid_gauss_smooth = mid_filter(gauss_array)  # 中值滤波
    aver_gauss_smooth = aver_filter(gauss_array)
    SNR = 0.9  # 信噪比
    impulse_array = add_impulse_noise(img_array, SNR)  # 添加椒盐噪声
    aver_impulse_smooth = aver_filter(impulse_array)  # 均值滤波去噪
    mid_impulse_smooth = mid_filter(impulse_array)

    plt.subplot(2, 4, 1)
    # 原图
    plt.imshow(img, cmap='gray')
    plt.title('原图')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    # 加高斯噪声
    plt.imshow(Image.fromarray(gauss_array), cmap='gray')
    plt.title('加高斯噪声')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    # 中值滤波去高斯噪声
    plt.imshow(Image.fromarray(mid_gauss_smooth))
    plt.title('中值滤波去高斯噪声')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    # 均值滤波去高斯噪声
    plt.imshow(Image.fromarray(aver_gauss_smooth))
    plt.title('均值滤波去高斯噪声')
    plt.axis('off')

    plt.subplot(2, 4, 5)
    # 原图
    plt.imshow(img, cmap='gray')
    plt.title('原图')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    # 加椒盐噪声
    plt.imshow(Image.fromarray(impulse_array), cmap='gray')
    plt.title('加椒盐噪声')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    # 中值滤波去椒盐噪声
    plt.imshow(Image.fromarray(mid_impulse_smooth))
    plt.title('中值滤波去椒盐噪声')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    # 均值滤波去椒盐噪声
    plt.imshow(Image.fromarray(aver_impulse_smooth))
    plt.title('均值滤波去椒盐噪声')
    plt.axis('off')

    plt.show()


def sobel(img_array, k):
    """Sobel算子边缘检测,3x3"""
    # input:img_array  数组  待检测轮廓的图像
    #           k      int   表示与像素点临近点的加权值

    kernel1 = np.array([[1, 0, -1], [k, 0, -k], [1, 0, -1]])  # 左边减右边
    kernel2 = np.array([[1, k, 1], [0, 0, 0], [-1, -k, -1]])  # 上边减下边
    m = img_array.shape[0]
    n = img_array.shape[1]
    rel_array = np.zeros((m - 2, n - 2))
    array1 = con(kernel1, img_array)
    array2 = con(kernel2, img_array)
    for i in range(m - 2):
        for j in range(n - 2):
            rel_array[i][j] = np.sqrt(array1[i][j] ** 2 + array2[i][j] ** 2)
    return rel_array


def prewitt(img_array):
    """prewitt算子边缘检测"""
    # input:img_array  数组  待检测轮廓的图像

    kernel1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # 右边减左边
    kernel2 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # 下边减上边
    m = img_array.shape[0]
    n = img_array.shape[1]
    rel_array = np.zeros((m - 2, n - 2))
    array1 = con(kernel1, img_array)
    array2 = con(kernel2, img_array)
    for i in range(m - 2):
        for j in range(n - 2):
            rel_array[i][j] = np.sqrt(array1[i][j] ** 2 + array2[i][j] ** 2)
    return rel_array


def roberts(img_array):
    """Roberts算子边缘检测，2x2"""

    m = img_array.shape[0]
    n = img_array.shape[1]
    rel_array = np.zeros((m - 1, n - 1))
    for i in range(m - 1):
        for j in range(n - 1):
            rel_array[i][j] = np.sqrt(
                (img_array[i][j] - img_array[i + 1][j + 1]) ** 2 + (img_array[i + 1][j] - img_array[i][j + 1]) ** 2)
    return rel_array


def gauss(x, y, sigma):
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


def canny(img_array, sigma):
    """canny算子边缘检测"""
    # 1.高斯滤波平滑图像
    # 2.用一阶偏导有限差分计算梯度幅值和方向
    # 3.对梯度幅值进行非极大值抑制
    # 4.用双阈值算法检测和连接边缘

    kernel = np.array([[gauss(0, 0, sigma), gauss(0, 1, sigma), gauss(0, 2, sigma)],
                       [gauss(1, 0, sigma), gauss(1, 1, sigma), gauss(1, 2, sigma)],
                       [gauss(2, 0, sigma), gauss(2, 1, sigma), gauss(2, 2, sigma)]])
    sum_kernel = np.sum(kernel)
    new_gray = con(kernel, img_array) / sum_kernel  # 高斯平滑
    # print(new_gray)
    # step2.增强 通过求梯度幅值
    w, h = new_gray.shape
    dx = np.zeros([w - 1, h - 1])
    dy = np.zeros([w - 1, h - 1])
    d = np.zeros([w - 1, h - 1])
    for i in range(w - 1):
        for j in range(h - 1):
            dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 图像梯度幅值作为图像强度值

    # setp3.非极大值抑制 NMS
    w2, h2 = d.shape
    NMS = np.copy(d)
    NMS[0, :] = NMS[w2 - 1, :] = NMS[:, 0] = NMS[:, h2 - 1] = 0
    for i in range(1, w2 - 1):
        for j in range(1, h2 - 1):
            if d[i, j] == 0:
                NMS[i, j] = 0
            else:
                grad_x = dx[i, j]
                grad_y = dy[i, j]
                gradTemp = d[i, j]

                # 如果Y方向幅度值较大
                if np.abs(grad_y) > np.abs(grad_x):
                    weight = np.abs(grad_x) / np.abs(grad_y)
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]
                    # 如果x,y方向梯度符号相同
                    if grad_x * grad_y > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]

                # 如果X方向幅度值较大
                else:
                    weight = np.abs(grad_y) / np.abs(grad_x)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]
                    # 如果x,y方向梯度符号相同
                    if grad_x * grad_y > 0:
                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                grad_temp1 = weight * grad1 + (1 - weight) * grad2
                grad_temp2 = weight * grad3 + (1 - weight) * grad4
                if gradTemp >= grad_temp1 and gradTemp >= grad_temp2:
                    NMS[i, j] = gradTemp
                else:
                    NMS[i, j] = 0

    # step4. 双阈值算法检测、连接边缘
    w3, h3 = NMS.shape
    rel_array = np.zeros([w3, h3])
    # 定义高低阈值
    tl = 0.15 * np.max(NMS)
    th = 0.3 * np.max(NMS)
    for i in range(1, w3 - 1):
        for j in range(1, h3 - 1):
            if (NMS[i, j] < tl):
                rel_array[i, j] = 0
            elif (NMS[i, j] > th):
                rel_array[i, j] = 1
            elif ((NMS[i - 1, j - 1:j + 1] < th).any() or (NMS[i + 1, j - 1:j + 1]).any()
                  or (NMS[i, [j - 1, j + 1]] < th).any()):
                rel_array[i, j] = 1

    # DT = cv2.bitwise_not(DT)
    return rel_array


def edge_detect():
    """第三部分：边缘检测"""
    # img = Image.open('lina.jpg').convert('L')  # 读入图片，图片的尺寸为128x128
    r, g, b = bmp_read('lena.bmp')  # 得到三个通道
    img = 0.2999 * r + 0.587 * g + 0.114 * b  # 改为单通道

    img_array = np.array(img).astype('float')  # 转化为array
    # Sobel算子
    k1 = 1
    k2 = 2
    sobel_array_1 = sobel(img_array, k1) / 2
    sobel_array_2 = sobel(img_array, k2) / 2
    # Roberts算子
    roberts_array = roberts(img_array)
    # Premitt算子
    prewitt_array = prewitt(img_array) / 2
    # canny算子
    sigma = 0.5
    canny_array = canny(img_array, sigma)

    plt.figure()

    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("原图")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(Image.fromarray(sobel_array_1))
    plt.title("sobel 3x3,k=" + str(k1))
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(Image.fromarray(sobel_array_2))
    plt.title("sobel 3x3,k=" + str(k2))
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(Image.fromarray(roberts_array))
    plt.title("roberts")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(Image.fromarray(prewitt_array))
    plt.title("premitt")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(canny_array, cmap='gray')  # 注：如果图像已经进行二值化以后就不需要使用fromarray来显示图像了
    plt.title("canny,sigma=" + str(sigma))
    plt.axis("off")

    plt.show()


def test_smooth():
    """第二部分：图像平滑"""
    img = Image.open('test.jpg').convert('L')  # 读入图片，图片的尺寸为128x128
    img_array = np.array(img)  # 转化为array

    mid_gauss_smooth = mid_filter(img_array)  # 中值滤波
    aver_gauss_smooth = aver_filter(img_array)

    plt.subplot(2, 3, 1)
    # 原图
    plt.imshow(img, cmap='gray')
    plt.title('原图')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    # 中值滤波去高斯噪声
    plt.imshow(Image.fromarray(mid_gauss_smooth))
    plt.title('中值滤波去噪声')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    # 均值滤波去高斯噪声
    plt.imshow(Image.fromarray(aver_gauss_smooth))
    plt.title('均值滤波去噪声')
    plt.axis('off')

    plt.show()


def bmp_read(filepath):
    """BMP格式读取图像"""
    file = open(filepath, "rb")
    file.read(18)  # 跳过一些信息
    biwidth = unpack("<i", file.read(4))[0]  # 图像的宽度 单位 像素
    biheight = unpack("<i", file.read(4))[0]  # 图像的高度 单位 像素
    file.read(2)  # 跳过一些信息
    bibitcount = unpack("<h", file.read(2))[0]  # 说明比特数
    file.read(24)  # 跳过一些信息
    bmp_data = []

    if bibitcount != 24:
        print("输入的图片比特值为 ：" + str(bibitcount) + "\t 与程序不匹配")

    for height in range(biheight):
        bmp_data_row = []
        # 四字节填充位检测
        count = 0
        for width in range(biwidth):
            bmp_data_row.append(
                [unpack("<B", file.read(1))[0], unpack("<B", file.read(1))[0], unpack("<B", file.read(1))[0]])
            count = count + 3
        # bmp 四字节对齐原则
        while count % 4 != 0:
            file.read(1)
            count = count + 1
        bmp_data.append(bmp_data_row)
    bmp_data.reverse()
    file.close()
    # R, G, B 三个通道
    r = []
    g = []
    b = []

    for row in range(biheight):
        r_row = []
        g_row = []
        b_row = []
        for col in range(biwidth):
            b_row.append(bmp_data[row][col][0])
            g_row.append(bmp_data[row][col][1])
            r_row.append(bmp_data[row][col][2])
        b.append(b_row)
        g.append(g_row)
        r.append(r_row)
    b = np.array(b)
    g = np.array(g)
    r = np.array(r)
    return r, g, b


if __name__ == '__main__':
    # add_noise()  # 添加高斯噪声和椒盐噪声
    # smooth()  # 图像平滑
    # test_smooth()  # 图像平滑,别的图片测试
    edge_detect()  # 图像边缘检测

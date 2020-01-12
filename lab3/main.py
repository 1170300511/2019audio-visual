import struct


def read_mfcc_matrix(filename):
    """读取mfc特征文件的信息，便于计算DTW距离"""
    f = open(filename, "rb")
    nframes = struct.unpack(">i", f.read(4))[0]  # 采样数
    frate = struct.unpack(">i", f.read(4))[0]  # 100 ns 内的
    nbytes = struct.unpack(">h", f.read(2))[0]  # 特征的字节数
    feakind = struct.unpack(">h", f.read(2))[0]
    ndim = nbytes / 4  # 维数
    feature = []
    for m in range(nframes):
        feature_frame = []
        for n in range(int(ndim)):
            feature_frame.append(struct.unpack(">f", f.read(4))[0])
        feature.append(feature_frame)
    f.close()
    return feature  # 返回一个长为n的特征列表，列表中每个元素都是39维数据


def cal_distance(x1, x2):
    """计算两个维数相等的向量的欧式距离"""
    length = len(x1)
    sums = 0
    for i in range(length):
        sums += abs(x1[i] - x2[i])
    return sums


def dtw_distance(x, y):
    """计算两个采用的dtw距离"""
    # 如果两个向量等长，不需要对齐
    if len(x) == len(y):
        rel = 0
        for i in range(len(x)):
            rel += cal_distance(x[i], y[i])
        return rel
    else:
        len_x = len(x)
        len_y = len(y)
        cost = [[0 for i in range(len_y)] for i in range(len_x)]
        # 初始化 dis 数组
        dis = []
        for i in range(len_x):
            dis_row = []
            for j in range(len_y):
                dis_row.append(cal_distance(x[i], y[j]))
            dis.append(dis_row)
        # 初始化 cost 的第 0 行和第 0 列
        cost[0][0] = dis[0][0]
        for i in range(1, len_x):
            cost[i][0] = cost[i - 1][0] + dis[i][0]
        for j in range(1, len_y):
            cost[0][j] = cost[0][j - 1] + dis[0][j]
        # 开始动态规划
        for i in range(1, len_x):
            for j in range(1, len_y):
                cost[i][j] = min(cost[i - 1][j] + dis[i][j] * 1,
                                 cost[i - 1][j - 1] + dis[i][j] * 2,
                                 cost[i][j - 1] + dis[i][j] * 1)
        return cost[len_x - 1][len_y - 1]


if __name__ == '__main__':
    # 读入模板
    model_nums = 5  # 模板个数
    test_nums = 8  # 每个模板的测试数据个数
    model_array = []  # 存储模板
    for i in range(model_nums):
        # 依次读入模板文件，存入列表中
        filename = 'model' + str(i + 1) + '.mfc'
        model_array.append(read_mfcc_matrix(filename))
    # 读入测试
    tag = []  # 存储每个测试数据的类别
    test_array = []  # 存储测试数据
    for i in range(model_nums):
        for j in range(test_nums):
            filename = 'test' + str(i + 1) + str(j + 1) + '.mfc'
            tag.append(i)
            test_array.append(read_mfcc_matrix(filename))
    print(tag)
    # 使用dtw判断类别
    rel_tag = []
    for j in range(model_nums * test_nums):
        rec_num = -1
        rec_dis = float('inf')
        for i in range(model_nums):
            dis = dtw_distance(test_array[j], model_array[i])
            if dis < rec_dis:
                rec_num = i
                rec_dis = dis
        rel_tag.append(rec_num)
    print(rel_tag)
    # 统计正确率
    cnt = 0
    for i in range(model_nums * test_nums):
        if tag[i] == rel_tag[i]:
            cnt += 1
    print('the rate of correct:' + str(cnt * 1.0 / (model_nums * test_nums)))

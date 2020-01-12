from scipy.io import wavfile
import numpy as np
import math
import wave

"""
返回一个元组，第一项为音频的采样率
第二项为音频数据的numpy数组
"""


# 符号函数
def sig(x):
    if x >= 0:
        return 1
    else:
        return 0


class Sound:
    matrix = []
    frames = 0
    samples = 0
    WIN_SIZE = 256
    win = np.hamming(WIN_SIZE)  # 直角窗（方窗）

    def read(self, number):
        file_name = '.\\wavs\\' + str(number) + ".wav"
        fs, data = wavfile.read(file_name)
        self.frames = int(len(data) / 256)
        if len(data) % 256 != 0:
            self.frames += 1
        self.matrix = np.zeros((self.frames, 256))
        i = 0
        j = 0
        cnt = 0
        self.samples = len(data)
        while j < 266:
            self.matrix[i][j] = data[cnt]
            cnt += 1
            j += 1
            if j == 256:
                i += 1
                j = 0
            if cnt == self.samples:
                break

    # 计算第i帧的能量
    def cal_energy(self, i):
        energy = 0
        for j in range(self.WIN_SIZE):
            energy += (self.matrix[i][j] * self.win[j]) ** 2
        return energy

    # 计算第i帧的过零率
    def cal_pass_zeros(self, i):
        cnt = 0
        for j in range(1, self.WIN_SIZE):
            cnt += math.fabs(sig(self.matrix[i][j]) - sig(self.matrix[i][j - 1])) * self.win[j]
        return cnt / self.WIN_SIZE

    # 计算每帧的过零率和每帧的能量，并打印在相应的文件中
    def print_energy_and_pass_zeros(self, number):
        file_energy = str(number) + "_en.txt"
        file_zeros = str(number) + "_zero.txt"
        f_en = open(file_energy, "w")
        f_zeros = open(file_zeros, "w")
        for i in range(self.frames):
            en = self.cal_energy(i)
            f_en.write(str(en) + "\n")
            zeros = self.cal_pass_zeros(i)
            f_zeros.write(str(zeros) + "\n")
        f_en.close()
        f_zeros.close()

    """
    # 双门限法检测端点
    def two_limit_doors(self):
        ans = []
        mh = 0
        ml = 0
        zs = 0
        left = 0
        right = self.frames
        for j in range(self.frames):
            mh += self.cal_energy(j) / (self.frames * 5)
            zs += self.cal_pass_zeros(j) / (self.frames * 100)

        ml = mh / 20

        # 用mh过滤
        for j in range(self.frames):
            if self.cal_energy(j) > mh:
                left = j
                break
        for j in range(self.frames - 1, -1, -1):
            if self.cal_energy(j) > mh:
                right = j
                break

        # 用ml过滤
        for j in range(left, -1, -1):
            if self.cal_energy(j) < ml:
                left = j
                break
        for j in range(right, self.WIN_SIZE):
            if self.cal_energy(j) < ml:
                right = j
                break

        # 用zs过滤
        for j in range(left, -1, -1):
            if self.cal_pass_zeros(j) < zs:
                left = j
                break
        for j in range(right, self.frames):
            if self.cal_pass_zeros(j) < zs:
                right = j
                break
        ans.append(left)
        ans.append(right)
        return ans
    """

    # 过滤静音部分
    def remove_silence(self):
        aver_energy = 0
        aver_zeros = 0
        for j in range(self.frames):
            aver_energy += self.cal_energy(j) / self.frames
            aver_zeros += self.cal_pass_zeros(j) / self.frames
        aver_energy /= 5
        aver_zeros /= 2

        door = 5
        flag = 2
        cnt = 0
        dot = []

        for i in range(self.frames):
            dot.append(i)
            if self.cal_energy(i) < aver_energy or self.cal_pass_zeros(i) < aver_zeros:
                cnt += 1
            else:
                flag -= 1
                if flag == 0:
                    if cnt >= door:
                        for j in range(1, cnt + 1):
                            dot.pop()
                    cnt = 0
                    flag = 2
        for i in range(1, cnt + 1):
            dot.pop()
        return dot

    # 打印pcm文件
    def pcm(self, number):
        file_name = str(number) + ".pcm"
        f = wave.open(file_name, "wb")
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        points = self.remove_silence()
        ans = []
        for i in points:
            for j in range(self.WIN_SIZE):
                ans.append(self.matrix[i][j])
        f.writeframes(np.array(ans).astype(np.short).tostring())
        f.close()

# 打印过零率和能量文件
def energy_and_zeros():
    test = Sound()
    for number in range(1, 11):
        test.read(number)
        test.print_energy_and_pass_zeros(number)


# 输出pcm文件
def print_pcm():
    test = Sound()
    for number in range(1, 11):
        print(number)
        test.read(number)
        test.pcm(number)


print_pcm()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.fftpack import dct, idct
import numpy as np
import struct
import argparse

# -------------- 常量 --------------------

# 分块维度

BLOCK_SIZE = 8

# 量化表
q_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.int)




# BMP读取类
class BMP(object):

    # 读取文件头
    def ReadHeader(self,f):
        f.seek(10)
        self.offset = struct.unpack('<I', f.read(4))[0]
        f.seek(4,1)
        (self.width, self.height) = struct.unpack('<II', f.read(8))
        self.skip = (4 - self.width * 3 % 4) % 4
    
    # 读取RGB Plane信息，plane顺序为(B,G,R)
    def ReadRGB(self,f):
        f.seek(self.offset)
        self.Plane = [np.zeros((self.height, self.width), dtype=np.int)] * 3

        for row in reversed(range(self.height)):
            for col in range(self.width):
                self.Plane[0][row][col], self.Plane[1][row][col], self.Plane[2][row][col] = struct.unpack('<BBB', f.read(3))
            f.seek(self.skip,1)

    def ShowInfo(self):
        print("图像宽度",self.width)
        print('图像高度',self.height)
        print('位图数据偏移量',self.offset)
        print('对齐填充',self.skip)
    
    def __init__(self,filelocation):
        self.filelocation = filelocation
        with open(filelocation, 'rb') as f:
            self.ReadHeader(f)
            self.ReadRGB(f)

            # 填充8位对齐

            row_padding, col_padding = 8 - self.height % 8, 8 - self.width % 8

            for cnt in range(row_padding):
                for i in range(3):
                    self.Plane[i] = np.row_stack((self.Plane[i],[0] * self.width))
            self.height = self.height + row_padding

            for cnt in range(col_padding):
                for i in range(3):
                    self.Plane[i] = np.column_stack((self.Plane[i],[0] * self.height))
            self.width = self.width + col_padding



# DCT+量化
def DCT2(mat):
    return np.round(dct(dct(mat, norm='ortho').T, norm='ortho') // q_table)


# 把数字二进制位拆分成List
def ToBinList(digit):
    binary = '0' * (8-(len(bin(digit))-2)) + bin(digit).replace('0b', '')
    return list(map(int, list(binary)))


# 解码UTF-8
def BinaryToStr(binary):
    index = 0
    string = []
    rec = lambda x, i: x[2:8] + (rec(x[8:], i-1) if i > 1 else '') if x else ''
    fun = lambda x, i: x[i+1:8] + rec(x[8:], i-1)
    while index + 1 < len(binary):
        chartype = binary[index:].index('0')
        length = chartype*8 if chartype else 8
        string.append(chr(int(fun(binary[index:index+length], chartype), 2)))
        index += length
    return ''.join(string)

# 解码消息
def MsgDecode():
    return BinaryToStr("".join(map(str,decodeList)))


# 展示矩阵效果
def ShowDCT(after):
    print("*********************************")
    for row8 in range(bmp.height // BLOCK_SIZE):
        for col8 in range(bmp.width // BLOCK_SIZE):
            for c in range(3):
                if after:
                    print("【隐写后】 颜色通道:%d 位置:(%d,%d) 8*8 DCT系数矩阵为:" % (c,row8*8,col8*8))
                else:
                    print("颜色通道:%d 位置:(%d,%d) 8*8 DCT系数矩阵为:" % (c,row8*8,col8*8))
                print(dctMat[c][row8:row8+BLOCK_SIZE, col8:col8+BLOCK_SIZE])
    print("*********************************")

# 展示一个例子
def ShowExample():
    print("左上角 8*8 DCT矩阵 B通道")
    print("隐写前")
    print(DCT2(bmp.Plane[0][0:8, 0:8]))
    print("隐写后")
    print(dctMat[0][0:8, 0:8])

    print("左上角 8*8 DCT矩阵 G通道")
    print("隐写前")
    print(DCT2(bmp.Plane[1][0:8, 0:8]))
    print("隐写后")
    print(dctMat[1][0:8, 0:8])


    print("左上角 8*8 DCT矩阵 R通道")
    print("隐写前")
    print(DCT2(bmp.Plane[2][0:8, 0:8]))
    print("隐写后")
    print(dctMat[2][0:8, 0:8])

    print("隐写信息对应的二进制位为")
    print(msgList)
   


def Calc_DCT_And_Msg():
    global bmp, msg, dctMat, msgList


    # 生成DCT表
    for row8 in range(0, bmp.height, BLOCK_SIZE):
        for col8 in range(0, bmp.width, BLOCK_SIZE):
            for c in range(3):
                dctMat[c][row8:row8+BLOCK_SIZE, col8:col8+BLOCK_SIZE] = DCT2(bmp.Plane[c][row8:row8+BLOCK_SIZE, col8:col8+BLOCK_SIZE])


    print("量化后的DCT系数矩阵如下:")
    ShowDCT(False)
    msg2 = bytearray(msg,'UTF-8')
    # print(msg2)
    
    for d in msg2:
        msgList = msgList + ToBinList(d)
 


# Jsteg编码
def JstegEncode():
    global bmp, msg, dctMat, msgList
    Calc_DCT_And_Msg()

    cnt = 0

    global msgLen
    msgLen = len(msgList)

    for row in range(bmp.height):
        for col in range(bmp.width):
            for c in range(3):
                data = dctMat[c][row][col]
                if data == 0 or data == 1 or data == -1:
                    continue
                if data%2 != msgList[cnt]:
                    dctMat[c][row][col] = (1 if data > 0 else -1) * (abs(data) -  (abs(data) % 2)  + msgList[cnt])
                cnt = cnt + 1
                if cnt == msgLen:
                    print("Jsteg隐写后的DCT系数矩阵如下:")
                    ShowDCT(True)
                    return True
    return False

# Jsteg解码
def JstegDecode():
    global decodeList
    decodeList = []
    cnt = 0
    for row in range(bmp.height):
        for col in range(bmp.width):
            for c in range(3):
                data = dctMat[c][row][col]
                if data == 0 or data == 1 or data == -1:
                    continue
                decodeList = decodeList + [data % 2]
                cnt = cnt + 1
                if cnt == msgLen:
                    return True
    return False


# F3 编码
def F3Encode():
    global bmp, msg, dctMat, msgList
    Calc_DCT_And_Msg()

    cnt = 0

    global msgLen
    msgLen = len(msgList)

    for row in range(bmp.height):
        for col in range(bmp.width):
            for c in range(3):
                data = dctMat[c][row][col]
                if data == 0:
                    continue
                if (data == -1 or data == 1) and msgList[cnt] == 0:
                    dctMat[c][row][col] = 0
                    continue
                
                if data%2 != msgList[cnt]:
                    dctMat[c][row][col] = (1 if data > 0 else -1) * (abs(data) - 1)
                cnt = cnt + 1
                if cnt == msgLen:
                    print("F3隐写后的DCT系数矩阵如下:")
                    ShowDCT(True)
                    return True
    return False

# F3 解码
def F3Decode():
    global decodeList
    decodeList = []
    cnt = 0
    for row in range(bmp.height):
        for col in range(bmp.width):
            for c in range(3):
                data = dctMat[c][row][col]
                if data == 0:
                    continue
                decodeList = decodeList + [data % 2]
                cnt = cnt + 1
                if cnt == msgLen:
                    return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Jsteg&&F3隐写程序 @2017-03-13")
    parser.add_argument('--src', action = "store", dest = "src", help='BMP源文件路径',required = True)
    parser.add_argument('--msg', action = "store", dest = "msg", help='隐写数据信息',required = True)
    parser.add_argument('--func', action = "store", dest = "func", help='隐写方式选择',required = True)

    given_args = parser.parse_args()
    src = given_args.src
    msg = given_args.msg
    func = given_args.func

    # 读取文件
    bmp = BMP(src)

    # 量化后的DCT总表
    dctMat = [np.zeros((bmp.height, bmp.width), dtype=np.int), np.zeros((bmp.height, bmp.width), dtype=np.int), np.zeros((bmp.height, bmp.width), dtype=np.int)]

    # msg编码List
    msgList = []
    decodeList = []
    

    # 选择编码方式
    if func == "js":
        JstegEncode()
        JstegDecode()
        ShowExample()
        print("****************************")
        print("Jsteg解码后读取的隐写信息为")
        print(MsgDecode())
    elif func == 'f3':
        F3Encode()
        F3Decode()
        ShowExample()
        print("****************************")
        print("F3解码后读取的隐写信息为")
        print(MsgDecode())



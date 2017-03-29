#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse

BLOCK_SIZE = 8
PIC_FILE = 'lena.jpg'

'''
    [1,  3,  6,  10, 15, 21, 28, 36],
    [2,  5,  9,  14, 20, 27, 35, 43],
    [4,  8,  13, 19, 26, 34, 42, 49],
    [7,  12, 18, 25, 33, 41, 48, 54],
    [11, 17, 24, 32, 40, 47, 53, 58], 
    [16, 23, 31, 39, 46, 52, 57, 61],
    [22, 30, 38, 45, 51, 56, 60, 63],
    [29, 37, 44, 50, 55, 59, 62, 64]
'''
# zigzag 顺序
zigzag = [(s-x,x) for s in range(8) for x in range(s+1)] + [(x,8+s-x) for s in range(7) for x in range(7,s,-1)]

# 计算像素相关性
def calcRelationship(subMat,color):
    r = 0
    for idx in range(len(zigzag)-1):
        x,y = zigzag[idx]
        nx,ny = zigzag[idx+1]
        r = r + abs(int(subMat[nx][ny][color]) - int(subMat[x][y][color]))
    return r
        
# F +1
def F_Positive(subMat,color):
    afterMat = subMat.copy()
    for x in range(8):
        for y in range(8):
            afterMat[x][y] = afterMat[x][y] ^ 1
    return calcRelationship(afterMat,color)

# F -1
def F_Negative(subMat,color):
    afterMat = subMat.copy()
    for x in range(8):
        for y in range(8):
            afterMat[x][y] = ((afterMat[x][y] + 1) ^ 1) - 1
    return calcRelationship(afterMat,color)

# 展示柱状图
def show():
    data = [Rpm,Rnm,Spm,Snm]
    plt.bar([0,1,3,4],data,width = 0.9, color='bbrr', tick_label = ['R m','R -m','S m','S -m'])
    plt.show()


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "RS隐写分析程序 @ 2017-03-29")
    parser.add_argument('--src', action = "store", dest = "src", help='分析图片路径',required = True)
    given_args = parser.parse_args()
    src = given_args.src
    im = Image.open(src)
    size_w,size_h = im.size
    print("width =",size_w,"height =",size_h)
    pixelsMat = np.array(im)

    original = []
    Fp_result = []
    Fn_result = []

    for color in range(3):
        for row in range(size_h // BLOCK_SIZE):
            for col in range(size_w // BLOCK_SIZE):
                subMat = pixelsMat[row*8:row*8+8, col*8:col*8+8]
                # 存储各块的像素相关度
                original.append(calcRelationship(subMat,color))
                Fp_result.append(F_Positive(subMat, color))
                Fn_result.append(F_Negative(subMat,color))



    Rpm = 0
    Spm = 0
    Rnm = 0
    Snm = 0

    total = len(original)

    for i in range(total):
        if Fp_result[i] > original[i]:
            Rpm = Rpm + 1
        else:
            Spm = Spm + 1
        
        if Fn_result[i] > original[i]:
            Rnm = Rnm + 1
        else:
            Snm = Snm + 1

    Rpm = Rpm / total
    Spm = Spm / total
    Rnm = Rnm / total
    Snm = Snm / total


    print('There are',total,'blocks in all.')
    print('R m = ',Rpm,'...R -m =',Rnm)
    print('S m = ',Spm,'...S -m =',Snm)
    show()
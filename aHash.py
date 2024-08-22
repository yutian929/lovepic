import cv2
import  numpy as np
import os
def aHash(img):# 均值哈希算法
    #缩放为8*8
    img=cv2.resize(img,(10,10),interpolation=cv2.INTER_CUBIC)
    #转换为灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    #遍历累加求像素和
    for i in range(10):
        for j in range(10):
            s=s+gray[i,j]
    #求平均灰度
    avg=s/64
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(10):
        for j in range(10):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

def cmpHash(hash1,hash2):# 哈希值比较
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

current_dir = os.path.dirname(os.path.abspath(__file__))
lovepic_path = current_dir + "\\0.JPG"
img = cv2.imread(lovepic_path)

height = img.shape[0]
width = img.shape[1]
ROI_w = 20
ROI_h = 20
now_dir = os.getcwd()
SIMILAR = []

for row in range(0,int(height/ROI_h)):
    for col in range(0,int(width/ROI_w)):
        print("ROI=(", row, "/", int(height / ROI_h), ",", col, "/", int(width / ROI_w), ")")
        img_ROI = img[row * ROI_h:(row + 1) * ROI_h, col * ROI_w:(col + 1) * ROI_w]
        hash1 = aHash(img_ROI)
        # 已经选好了ROI，下面进行1500次匹配,并存储相应的相似度
        for index in range(1, 1501):
            pic_path = now_dir + "\\pics\\" + str(index) + ".JPG"
            pic = cv2.imread(pic_path)
            hash2 = aHash(pic)
            n = cmpHash(hash1, hash2)
            # print("pic = ", index, "similarity = ",n)
            SIMILAR.append(n)
        #找到相似度最大的图片下标
        most_index = SIMILAR.index(min(SIMILAR))
        print("choose pic-",most_index)
        SIMILAR.clear()
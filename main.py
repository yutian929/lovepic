import cv2
import os
import numpy as np

def channels_read(path):# 把每一个pic读进来并返回b,g,r的众数
    pic = cv2.imread(path)
    b, g, r = cv2.split(pic)

    b_flat = np.array(b).flatten()
    counts = np.bincount(b_flat)
    b_mode = np.argmax(counts)

    g_flat = np.array(g).flatten()
    counts = np.bincount(g_flat)
    g_mode = np.argmax(counts)

    r_flat = np.array(r).flatten()
    counts = np.bincount(r_flat)
    r_mode = np.argmax(counts)

    return b_mode,g_mode,r_mode

def aHash(img):# 均值哈希算法
    #缩放为8*8
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    #转换为灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    #遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
    #求平均灰度
    avg=s/64
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str


# 读取lovepic
current_dir = os.path.dirname(os.path.abspath(__file__))
lovepic_path = current_dir + "\\0.JPG"
img = cv2.imread(lovepic_path)

# 读取图片的高和宽
height = img.shape[0]
width = img.shape[1]

# 1.先将图片分成若干个20x20的ROI
ROI_w = 20
ROI_h = 20
B = []
G = []
R = []
for row in range(0,int(height/ROI_h)):
    for col in range(0,int(width/ROI_w)):
        img_ROI = img[row*ROI_h:(row+1)*ROI_h,col*ROI_w:(col+1)*ROI_w]
        # cv2.imshow('roi',img_ROI)
        # cv2.waitKey(0)
        # 2.把每个ROI三通道b,g,r的众数求出来并用B，G，R存储
        b,g,r = cv2.split(img_ROI)

        b_flat = np.array(b).flatten()
        counts = np.bincount(b_flat)
        B.append(np.argmax(counts))

        g_flat = np.array(g).flatten()
        counts = np.bincount(g_flat)
        G.append(np.argmax(counts))

        r_flat = np.array(r).flatten()
        counts = np.bincount(r_flat)
        R.append(np.argmax(counts))

# 3.方法一：用每一个ROI去匹配pics，每个ROI进行1500次三通道众数匹配，用相似度最大的拼接上去。
# 方法二：匹配每一个像素点，找相似度最高的拼接上去
# 方法三：哈希均值算法
# 方法四：直方图算法(selected)
now_dir = os.getcwd()
SIMILAR = []
for row in range(0,int(height/ROI_h)):
    for col in range(0,int(width/ROI_w)):
        print("ROI=(", row, "/", int(height / ROI_h), ",", col, "/", int(width / ROI_w), ")")
        img_ROI = img[row * ROI_h:(row + 1) * ROI_h, col * ROI_w:(col + 1) * ROI_w]
        # 计算图img_ROI的三通道直方图
        H0 = cv2.calcHist([img_ROI], [0], None, [256], [0, 256])
        H0 = cv2.normalize(H0, H0, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
        H1 = cv2.calcHist([img_ROI], [1], None, [256], [0, 256])
        H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
        H2 = cv2.calcHist([img_ROI], [2], None, [256], [0, 256])
        H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
        # 已经选好了ROI，下面进行1500次匹配,并存储相应的相似度
        for index in range(1, 1501):
            pic_path = now_dir + "\\pics\\" + str(index) + ".JPG"
            pic = cv2.imread(pic_path)
            # 计算图pic的直方图
            picH0 = cv2.calcHist([pic], [0], None, [256], [0, 256])
            picH0 = cv2.normalize(picH0, picH0, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
            picH1 = cv2.calcHist([pic], [1], None, [256], [0, 256])
            picH1 = cv2.normalize(picH1, picH1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
            picH2 = cv2.calcHist([pic], [2], None, [256], [0, 256])
            picH2 = cv2.normalize(picH2, picH2, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
            # 利用compareHist（）进行比较相似度
            similarity0 = cv2.compareHist(H0, picH0, 0)
            similarity1 = cv2.compareHist(H1, picH1, 0)
            similarity2 = cv2.compareHist(H2, picH2, 0)
            SIMILAR.append(similarity0+similarity1+similarity2)
        #选取最合适一张20x20的pic拼接上去，并且把SIMILAR清零
        most_index = SIMILAR.index(max(SIMILAR))
        most_path =  now_dir + "\\pics\\" + str(most_index) + ".JPG"
        SIMILAR.clear()
        # 读取该图片并进行融合/掩膜
        most_pic = cv2.imread(most_path)
        combine = cv2.addWeighted(most_pic,0.5,img[row*ROI_h:(row+1)*ROI_h,col*ROI_w:(col+1)*ROI_w],0.5,0)
        img[row*ROI_h:(row+1)*ROI_h,col*ROI_w:(col+1)*ROI_w] = combine
        print("combine successfully,choose pic---",most_index)
cv2.imshow('img_combine',img)
cv2.waitKey(0)
cv2.imwrite('lovepic.JPG',img)

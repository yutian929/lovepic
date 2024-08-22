import cv2
import  numpy as np
import os


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
        # 计算图img_ROI的直方图
        H1 = cv2.calcHist([img_ROI], [1], None, [256], [0, 256])
        H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
        # 已经选好了ROI，下面进行1500次匹配,并存储相应的相似度
        for index in range(1, 1501):
            pic_path = now_dir + "\\pics\\" + str(index) + ".JPG"
            pic = cv2.imread(pic_path)
            # 计算图pic的直方图
            H2 = cv2.calcHist([pic], [1], None, [256], [0, 256])
            H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)
            # 利用compareHist（）进行比较相似度
            similarity = cv2.compareHist(H1, H2, 0)
            SIMILAR.append(similarity)
        #找到相似度最大的图片下标
        most_index = SIMILAR.index(max(SIMILAR))
        print("choose pic-",most_index)
        SIMILAR.clear()
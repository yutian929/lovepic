import cv2
import os
import numpy as np

# 把每一个pic读进来并返回b,g,r的众数
def channels_read(path):
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

now_dir = os.getcwd()
print(now_dir)
for index in range(1,1501):
    pic_path = now_dir + "\\pics\\" + str(index) + ".JPG"
    pic_bgr = channels_read(pic_path)
    print(pic_bgr)
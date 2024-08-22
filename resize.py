import cv2
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
pics_dir = current_dir + "\pics"
for filename in os.listdir(pics_dir):
        print(filename)
        img = cv2.imread(pics_dir + "\\" + filename)
        img_resize = cv2.resize(img,(20,20))
        cv2.imwrite(pics_dir + "\\" + filename,img_resize)
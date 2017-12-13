from helpers import *
import cv2
import glob

img_fname = '../CND/racecar_data/test/imgs/1512969369245619058_1_-0.0_0.0.jpg'
imgs_path = '../CND/racecar_data/imgs/'
save_path = '../CND/racecar_data/imgs/crop/'
filenames = glob.glob(imgs_path + "/*.jpg")

for f in filenames:
    img = cv2.imread(img_fname, cv2.COLOR_BGR2RGB)
    img = img[240:720,:,:]
    img = cv2.resize(img, (320,160))
    fname = f.strip().lstrip(imgs_path)
    cv2.imwrite(save_path + fname, img)

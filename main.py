import numpy as np
from segment import segment
import cv2
from os.path import join
import os

def get_filenames(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir,name)) and (name.endswith(".bmp") or name.endswith(".jpg"))]

dataroot = '/media/bek/data/course/biometrics/iris_segmentation/db'
outputroot = dataroot.replace('/db','/output')
if not os.path.exists(outputroot):
        os.makedirs(outputroot)

filenames = get_filenames(dataroot)

for fn in filenames:
    out_file = join(outputroot,fn)
    in_file = join(dataroot,fn)
    im = cv2.imread(in_file,0)

    ciriris,cirpupil,imwithnoise = segment(im)
    cv2.circle(im,(ciriris[1],ciriris[0]),ciriris[2],(0,0,255),2)
    cv2.circle(im,(cirpupil[1],cirpupil[0]),cirpupil[2],(0,255,0),2)
    cv2.imwrite(out_file,im)



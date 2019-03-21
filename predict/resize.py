
import cv2
import os
import numpy as np


def resize(path_image, src_path, image_size):
    filelist = os.listdir(path_image)     # name+jpg
    for files in filelist:      # files=name+jpg
        fmat = os.path.splitext(files)[1]
        if str(fmat) == '.png' or str(fmat) == '.jpg':
            filename = os.path.splitext(files)[0]
        
            filepath = path_image+files
            # print filepath
            output_src_name = filename + '_color.png'
            # print filepath

            img =cv2.imread(filepath)

            h = img.shape[0]

            if h != image_size:
                img = cv2.resize(img, (image_size, image_size))

            cv2.imwrite(src_path + output_src_name, img)
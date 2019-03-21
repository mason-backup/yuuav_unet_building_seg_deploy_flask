import cv2
import os
import numpy as np

path_image = "./out_image/"
filelist = os.listdir(path_image)
for files in filelist:  # files=name+jpg
    filename = os.path.splitext(files)[0]
    print filename
    filepath = path_image + files
    print filepath

    try:
        img = cv2.imread(filepath, -1)

    except Exception:
        logging.info('The filepath (image) is not found!')
        raise

    h, w= img.shape
    img_color = np.zeros((h,w,3),dtype=np.uint8)
    for i in range(h):
        for j in range (w):
            if img[i, j] == 0:
                img_color[i, j, :] = [0, 128, 192]  # in[B,G,R] brown
            else:
                img_color[i, j, :] = [0, 0, 0]
    cv2.imwrite('./color_output/' + files, img_color)

# 
import cv2
import os
import numpy as np
import copy
import sys


def mask_overlap(image, mask_, color_bgr):
    mask_ = np.dstack((mask_, mask_, mask_))

    index = mask_[:, :, 0] == 0

    pct = 0
    for i in range(0, mask_.shape[0]):
        for j in range(0, mask_.shape[1]):
            if mask_[i, j, 0] == 0:
                pct = pct + 1
    building_pct = (pct * 100) / (mask_.shape[0]*mask_.shape[1])

    # print "the percentage is {0}% of total pixel {1}".format(building_pct, (mask_.shape[0] * mask_.shape[1]))

    mask_ = np.int64(mask_ < 1)
    mask_ = mask_*np.array(color_bgr)

    mask_ = mask_.astype(np.uint8)
    weight_coeffi = cv2.addWeighted(mask_, 0.8, image, 0.2, 0.)
    # cv2.imwrite('weight.png', weight_coeffi)

    image_ = copy.copy(image)

    image_[index] = weight_coeffi[index]

    return image_, building_pct


def main(path_image, path_src, path_):
    filelist = os.listdir(path_image)
    for files in filelist:  # files=name+jpg
        fmat=os.path.splitext(files)[1]
        if str(fmat) == '.png' or str(fmat) == '.jpg':
            filename = os.path.splitext(files)[0]
            filepath = path_image + files

            name = filename.split("%")[0]
            src_name = name + '_color.png'
            src_path = path_src + src_name

            try:
                mask = cv2.imread(filepath, -1)

                src = cv2.imread(src_path)
            except Exception:
                print ('The filepath (image) is not found!')
                raise
            image_size = mask.shape[0]
            # src = cv2.resize(src, (image_size, image_size)). # make sure the src is image_size
            img_mask, building_area = mask_overlap(src, mask, color_bgr=[0, 128, 192])

            cv2.imwrite(path_ + src_name, img_mask)

            # percentage = int(round(percentage))
            # sys.exit(percentage)
            return building_area


if __name__ == "__main__":
    path_image = "./visualization/out_image/"
    path_src = "./visualization/src/"
    path_ = "./results/"
    main()

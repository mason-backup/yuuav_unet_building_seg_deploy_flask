# coding:utf-8
import os, sys
import shutil

import resize
from Unet_predict.data import dataset_to_tfrecords
from Unet_predict import predict
from visualization import overlap_mask


def main():
    print "\n%%%%%%%%%%%%%%%%%%%%% 1) resize the test images"
    print os.getcwd()
    shutil.rmtree('./predict/visualization/src/')
    os.mkdir('./predict/visualization/src/')

    path_image = "./images/"
    src_path = "./predict/visualization/src/"
    image_size = 224
    resize.resize(path_image, src_path, image_size)

    print "\n%%%%%%%%%%%%%%%%%%%%% 2) dataset to tfrecords"
    shutil.rmtree('./predict/Unet_predict/tfrecords/')
    os.mkdir('./predict/Unet_predict/tfrecords/')

    tf_output_dir_ = "./predict/Unet_predict/tfrecords/"
    data_dir_ = "./predict/visualization/src/"
    name_color_ = "_color"
    dataset_to_tfrecords.main(tf_output_dir_, data_dir_, name_color_)

    print "\n%%%%%%%%%%%%%%%%%%%%% 3) predict"
    shutil.rmtree('./predict/visualization/out_image/')
    os.mkdir('./predict/visualization/out_image/')

    checkpoint_path = "./predict/Unet_predict/checkpoints/unet.ckpt-20000"
    tfrecords_dir = "./predict/Unet_predict/tfrecords/src-00000-of-00001.tfrecords"
    predict_output_dir = "./predict/visualization/out_image/"
    predict.main(checkpoint_path, tfrecords_dir, image_size, predict_output_dir)

    print "\n%%%%%%%%%%%%%%%%%%%%% 4) visualization"
    shutil.rmtree('./results/')
    os.mkdir('./results/')

    path_image = "./predict/visualization/out_image/"
    path_src = "./predict/visualization/src/"
    path_ = "./results/"
    predict_pct = overlap_mask.main(path_image, path_src, path_)

    return predict_pct

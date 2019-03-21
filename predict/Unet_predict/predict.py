# ============================================================== #
#                         Fusnet eval                            #
#                                                                #
#                                                                #
# Eval fusnet with processed dataset in tfrecords format         #
#                                                                #
# Author: Karim Tarek                                            #
# ============================================================== #

from __future__ import print_function

import tensorflow as tf
from PIL import Image

import os
import glob
import argparse

import data.dataset_loader as dataset_loader
import unet
import numpy as np

# Basic model parameters as external flags.
FLAGS = None


def load_datafiles(tfrecords_dir):
    """
    Get all tfrecords from tfrecords dir:
    """

    # tf_record_pattern = os.path.join(FLAGS.tfrecords_dir, '%s-*' % type)
    tf_record_pattern = tfrecords_dir
    data_files = tf.gfile.Glob(tf_record_pattern)

    data_size = 0
    for fn in data_files:
        for record in tf.python_io.tf_record_iterator(fn):
            data_size += 1

    return data_files, data_size


def maybe_save_images(images, filenames, output_dir):
    """
    Save images to disk
    -------------
    Args:
        images: numpy array     [batch_size, image_size, image_size]
        filenames: numpy string array, filenames corresponding to the images   [batch_size]
    """

    if output_dir is not None:
        batch_size = images.shape[0]
        for i in xrange(batch_size):
            image_array = images[i, :, :]
            file_path = os.path.join(output_dir, filenames[i])
            image = Image.fromarray(np.uint8(image_array))
            image.save(file_path)


def evaluate(checkpoint_path, tfrecords_dir, image_size, output_dir):

    data_files, data_size = load_datafiles(tfrecords_dir)
    images, filenames = dataset_loader.inputs(
                                    data_files = data_files,
                                    image_size = image_size,
                                    batch_size = 1,
                                    num_epochs = 1,
                                    train = False)
    # labels = tf.stack([labels, labels], 3)
    # labels = tf.reshape(labels, (FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 2))

    logits = unet.build(images, 2, False)

    predicted_images = unet.predict(logits, 1, image_size)

    # accuracy = unet.accuracy(logits, labels)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()

    sess.run(init_op)

    saver = tf.train.Saver()

    # if not tf.gfile.Exists(FLAGS.checkpoint_path + '.meta'):
    if not tf.gfile.Exists(checkpoint_path + '.meta'):
        raise ValueError("Can't find checkpoint file")
    else:
        print('[INFO    ]\tFound checkpoint file, restoring model.')
        saver.restore(sess, checkpoint_path)
    
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # global_accuracy = 0.0

    step = 0
    
    try:
    
        while not coord.should_stop():
            predicted_images_value, filenames_value = sess.run([predicted_images, filenames])
            # global_accuracy += acc_seg_value

            maybe_save_images(predicted_images_value, filenames_value, output_dir)
            # print('[PROGRESS]\tAccuracy for current batch: %.5f' % (acc_seg_value))
            step += 1

    except tf.errors.OutOfRangeError:
        print('[INFO    ]\tDone evaluating in %d steps.' % step)

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # global_accuracy = global_accuracy / step

    # print('[RESULT  ]\tGlobal accuracy = %.5f' % (global_accuracy))

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


def main(checkpoint_path, tfrecords_dir, image_size, output_dir):
    """
    Run unet prediction on input tfrecords
    """

    if output_dir is not None:
        if not tf.gfile.Exists(output_dir):
            print('[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(output_dir))
            tf.gfile.MakeDirs(output_dir)
        
    evaluate(checkpoint_path, tfrecords_dir, image_size, output_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Eval Unet on given tfrecords.')
    parser.add_argument('--tfrecords_dir', help = 'Tfrecords directory')
    parser.add_argument('--checkpoint_path', help = 'Path of checkpoint to restore. (Ex: ../Datasets/checkpoints/unet.ckpt-80000)')
    parser.add_argument('--num_classes', help = 'Number of segmentation labels', type = int, default = 2)
    parser.add_argument('--image_size', help = 'Target image size (resize)', type = int, default = 224)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 1)
    parser.add_argument('--output_dir', help = 'Output directory for the prediction files. If this is not set then predictions will not be saved')
    
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()

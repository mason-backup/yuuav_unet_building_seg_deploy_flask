# ============================================================== #
#             Dataset to tfrecords (open pipeline)               #
#                                                                #
#                                                                #
# Converts image data to TFRecords format with Example protos    #
# The image data set is expected to reside in img files located  #
# in the following structure data_dir/image_color.png...         #
#                                                                #
# Author: Karim Tarek                                            #
# ============================================================== #

from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os
import random
import sys
import threading

from image_coder import ImageCoder


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example(filename, image_buffer, height, width):
    image_format = 'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/filename': bytes_feature(tf.compat.as_bytes(os.path.basename(os.path.normpath(filename)))),
            'image/format': bytes_feature(tf.compat.as_bytes(image_format)),
            'image/encoded/color': bytes_feature(tf.compat.as_bytes(image_buffer))
            }))
    return example


def process_image(filename, coder, name_color):

    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()

    # Convert any JPEG to PNG for consistency
    if coder.is_jpeg(filename):
        print('[PROGRESS]\tConverting JPEG to PNG for %s' % filename)
        image_data = coder.jpeg_to_png(image_data)

    # Decode the PNG
    image = coder.decode_png(image_data)
    # image = image[:, :, 0:3]

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]

    # Check that color has 3 channels while depth, label just 1
    if name_color in filename:
        # print (image.shape[2])
        assert image.shape[2] == 3
    else:
        assert image.shape[2] == 1

    return image_data, height, width


def process_image_files_batch(coder, thread_index, ranges, name, filenames, name_color, output_dir):
    num_threads = len(ranges)
    assert not 1 % 1
    num_shards_per_batch = int(1 / 1)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):

        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d.tfrecords' % (name, shard, 1)
        output_file = os.path.join(output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]

            # Concatenate naming conventions to get color, label true paths
            # color_file = "{0}%{1}".format(filename, FLAGS.name_color)
            color_file = filename % name_color
            # label_file = filename % FLAGS.name_label

            image_buffer, height, width = process_image(color_file, coder, name_color)
            # label_buffer, height, width = process_image(label_file, coder)

            # example = convert_to_example(filename, image_buffer, label_buffer, height, width)
            example = convert_to_example(filename, image_buffer, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

        writer.close()
        print('[THREAD %d]\tWrote %d images to %s' %
                    (thread_index, shard_counter, output_file))
        shard_counter = 0

    print('[THREAD %d]\tWrote %d images to %d shards.' %
                (thread_index, counter, num_files_in_thread))


def process_image_files(name, filenames, name_color, output_dir):
    spacing = np.linspace(0, len(filenames), 1 + 1).astype(np.int)
    ranges = []

    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('[PROGRESS]\tLaunching %d threads for spacings: %s' % (1, ranges))

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, name_color, output_dir)
        t = threading.Thread(target=process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('[INFO    ]\tFinished writing all %d images in data set.' % len(filenames))


def find_image_files(data_dir, name_color):
    print('[PROGRESS]\tDetermining list of input files from %s' % data_dir)

    filenames = []

    # Construct the list of image files
    color_file_path = os.path.join(data_dir, '*%s.*') % (name_color)

    color_files = tf.gfile.Glob(color_file_path)
    # label_files = tf.gfile.Glob(label_file_path)

    # assert len(color_files) == len(label_files)

    matching_files = [x.replace(name_color, '%s') for x in color_files]
    # where the 08%s.png <--> 08_color.png comes from, !!!!

    filenames.extend(matching_files)

    # Shuffle the ordering of all image files in order to guarantee randomness
    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]

    print('[INFO    ]\tFound %d images inside %s.' % (len(filenames), data_dir))

    return filenames


def process_dataset(name, data_dir, name_color, output_dir):
    filenames = find_image_files(data_dir, name_color)
    process_image_files(name, filenames, name_color, output_dir)


def main(output_dir, data_dir, name_color):
    print('[INFO    ]\tSaving results to %s' % output_dir)

    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    process_dataset(os.path.basename(os.path.normpath(data_dir)), data_dir, name_color, output_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Converts image data to TFRecords file format')
    parser.add_argument('--data_dir', help = 'Data directory', required = True)
    parser.add_argument('--output_dir', help = 'Output data directory', required = False)
    parser.add_argument('--num_threads', help = 'Number of threads', type = int, default = 1)
    parser.add_argument('--num_shards', help = 'Number of shards in training TFRecord files', type = int, default = 1)
    parser.add_argument('--name_color', help = 'Color images name format', default = '_color')
    # parser.add_argument('--name_label', help = 'Label images name format', default = '_label')
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()

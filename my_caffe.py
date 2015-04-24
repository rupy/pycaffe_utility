#!/usr/local/bin/python
# -*- coding: utf-8 -*-

__author__ = 'rupy'

import caffe
import numpy as np
import os
import sys
import glob
import logging

class MyCaffe:


    def __init__(self, model_file, pretrained_file, mean_file, labels=None):

        # log setting
        program = os.path.basename(type(self).__name__)
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

        # path/to/*.prototxt
        self.model_file = model_file
        # path/to/*.caffemodel
        self.pretrained_file = pretrained_file
        # path/to/*_mean.npy
        self.mean_file = mean_file

        self.mean = np.load(mean_file)
        self.image_dims = (256, 256)
        self.raw_scale = 255
        self.channel_swap = (2, 1, 0)
        self.gpu_flag = False

        self.labels = labels
        self.net = caffe.Classifier(
            model_file=self.model_file,
            pretrained_file=self.pretrained_file,
            image_dims=self.image_dims,
            mean=self.mean,
            input_scale=None,
            raw_scale=self.raw_scale,
            channel_swap=self.channel_swap,
            gpu=self.gpu_flag
        )

        self.inputs = None
        self.image_files = None

    def load_img_files(self, img_file_or_dir, file_num_limit=10):
        # load image
        if img_file_or_dir.endswith('npy'): #npy
            self.logger.info('loading npy file: ', img_file_or_dir)
            inputs = np.load(img_file_or_dir)
        if os.path.isdir(img_file_or_dir): # directory(maximum image num to be loaded is batch size)
            image_extentions = ('jpg', 'jpeg', 'png', 'bmp', 'gif')
            self.inputs = []
            self.img_files = []
            file_num = 0
            for img_file in glob.glob(img_file_or_dir + '/*'):
                if img_file.endswith(image_extentions):
                    if file_num >= file_num_limit:
                        break
                    file_num += 1
                    self.logger.info('loading image file: %s', img_file)
                    self.inputs.append(caffe.io.load_image(img_file))
                    self.img_files.append(img_file)
                else:
                    self.logger.info('skipping: %s', img_file)
        else: # an image file
            self.logger.info('loading image file: %s', img_file_or_dir)
            self.inputs = [caffe.io.load_image(img_file_or_dir)]
            self.img_files = [img_file_or_dir]
            print self.inputs.shape

    def predict_by_imagenet(self, category_file, over_sample=False, top_k=3):

        if not self.inputs or not self.img_files:
            raise Exception('You should run load_img_files(), first.')

        # predict
        # predict method returns probability score of category in each image.
        self.logger.info('predicting')
        predictions = self.net.predict(self.inputs,oversample=over_sample)

        # show result sorted by score
        categories = np.loadtxt(category_file, str, delimiter="\t")
        for img_file, prediction in zip(self.img_files, predictions):
            print 'img file:', img_file
            prediction = zip(prediction.tolist(), categories)
            prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
            for rank, (score, name) in enumerate(prediction[:top_k], start=1):
                print('#%d | %s | %4.1f%%' % (rank, name, score * 100))

    def get_features(self, data_dir, save_file,  layer_name='fc7', over_sample=False):
        # get structure of network
        n_batches, n_channels, height, width = self.net.blobs[layer_name].data.shape
        self.logger.info('layer: %s n_batches: %d n_channels: %d height:%d, width: %d',
                         layer_name, n_batches, n_channels, height, width)
        n_dim = np.prod(self.net.blobs[layer_name].data.shape[1:]) # feature dimension

        # read directories
        self.logger.info("use images in %s", data_dir)
        file_names = [ f for f in os.listdir(data_dir)]
        data_len = len(file_names)
        self.logger.info("data length: %s", data_len)

        all_features = np.array([]).reshape(0, n_dim) # empty matrix
        self.logger.info("begin extracting features")
        for start in xrange(0, data_len, n_batches):
            self.logger.info("progress: %d / %d", start, data_len)

            # read batch images
            end = start + n_batches
            if data_len <= end:
                end = data_len
            batch_files = file_names[start:end]
            batch_data = [caffe.io.load_image(os.path.join(data_dir, f)) for f in batch_files]

            # predict
            predictions = self.net.predict(batch_data, oversample=over_sample)

            # extruct features
            features = self.net.blobs[layer_name].data.reshape(n_batches, n_dim)[0:end - start]
            all_features = np.vstack([all_features, features])
            
        # scaled_feature = preprocessing.scale(flatten_feature)
        self.logger.info("saving file to  %s", save_file)
        np.save(save_file, all_features)
        return all_features

    def create_lmdb(self):
        pass

if __name__ == '__main__':

    logging.root.setLevel(level=logging.INFO)
    pass

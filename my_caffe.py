#!/usr/local/bin/python
# -*- coding: utf-8 -*-

__author__ = 'rupy'

import caffe
import numpy as np
import os
import sys
import glob
import logging
from sklearn import preprocessing
from sklearn.externals import joblib
from pprint import pprint

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

        self.labels = labels
        self.net = caffe.Classifier(
            model_file=self.model_file,
            pretrained_file=self.pretrained_file,
            image_dims=self.image_dims,
            mean=self.mean,
            input_scale=None,
            raw_scale=self.raw_scale,
            channel_swap=self.channel_swap
        )

        self.inputs = None
        self.image_files = None

    def load_img_files(self, img_file_or_dir, file_num_limit=10):
        # load image
        # if img_file_or_dir.endswith('npy'):
        #     self.logger.info('loading npy file: ', img_file_or_dir)
        #     inputs = np.load(img_file_or_dir)
        if os.path.isdir(img_file_or_dir):
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
        else:
            self.logger.info('loading image file: %s', img_file_or_dir)
            inputs = [caffe.io.load_image(img_file_or_dir)]
            self.img_files = [img_file_or_dir]

    def predict_by_imagenet(self, category_file, over_sample=False, top_k=3):

        if not self.inputs or not self.image_files:
            Exception('You should run load_img_files(), first.')

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

    def get_features(self, layer_name='fc7', over_sample=False):

        if not self.inputs or not self.image_files:
            Exception('You should run load_img_files(), first.')

        self.logger.info('predicting')
        predictions = self.net.predict(self.inputs,oversample=over_sample)

        feature = self.net.blobs[layer_name].data[4]
        flatten_feature = feature.flatten().tolist()
        # scaled_feature = preprocessing.scale(flatten_feature)
        return self.net.blobs[layer_name].data.shape

    def create_lmdb(self):
        pass

if __name__ == '__main__':

    logging.root.setLevel(level=logging.INFO)
    pass
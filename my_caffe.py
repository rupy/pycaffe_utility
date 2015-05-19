#!/usr/local/bin/python
# -*- coding: utf-8 -*-

__author__ = 'rupy'

import caffe
import numpy as np
import os
import sys
import glob
import logging
import matplotlib.pyplot as plt
from matplotlib import colors
from google.protobuf import text_format
from caffe.proto import caffe_pb2
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import itertools

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

        self.mean = np.load(mean_file).mean(1).mean(1)
        self.image_dims = None
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
        caffe.set_mode_gpu()

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

    def predict_by_imagenet(self, category_file=None, over_sample=False, top_k=3, plot_flag=False):

        if not self.inputs or not self.img_files:
            raise Exception('You should run load_img_files(), first.')

        # predict
        # predict method returns probability score of category in each image.
        self.logger.info('predicting')
        predictions = self.net.predict(self.inputs,oversample=over_sample)
        
        print colors.cnames.keys()
        if plot_flag:
            for i, prediction in enumerate(predictions):
                plt.subplot(len(predictions) / 4 + 1, 4, i)
                plt.plot(predictions[i], c=colors.cnames.keys()[i])
                print i
            plt.show()
        
        for prediction in predictions:
            print "predicted class num: %d" %  prediction.argmax()

        # show result sorted by score
        categories = np.loadtxt(category_file, str, delimiter="\t")
        for img_file, prediction in zip(self.img_files, predictions):
            print 'img file:', img_file
            prediction = zip(prediction.tolist(), categories)
            prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
            for rank, (score, name) in enumerate(prediction[:top_k], start=1):
                print('#%d | %s | %4.1f%%' % (rank, name, score * 100))

    def get_features(self, data_dir, save_file, cat_file,  layer_name='fc6wi', n_batches=None):

        self.logger.info("begin extracting features")
        # get structure of network
        n_inputs, n_channels = self.net.blobs[layer_name].data.shape
        if n_batches is None:
            n_batches = n_inputs

        self.logger.info('layer: %s n_inputs: %d n_batches: %d n_channels: %d', layer_name, n_inputs, n_batches, n_channels)
        n_dim = np.prod(self.net.blobs[layer_name].data.shape[1:]) # feature dimension

        # read categories
        self.logger.info("data dir is %s", data_dir)
        category_dirs = sorted([ f for f in os.listdir(data_dir)])

        all_features = np.array([]).reshape(0, n_dim) # empty matrix
        cat_list = []
        for i, category_dir in enumerate(category_dirs):

            # read data
            self.logger.info("category: <%s> (%d / %d)", category_dir, i + 1, len(category_dirs))
            category_path = os.path.join(data_dir, category_dir)
            file_names = sorted([f for f in os.listdir(category_path)])
            data_num = len(file_names)
            self.logger.info("data num: %s", data_num)
            cat_list.extend([i] * data_num)
            
            for start in xrange(0, data_num, n_batches):
                self.logger.info("progress: %d / %d", start + 1, data_num)
                
                # read batch images
                end = start + n_batches
                if data_num <= end:
                    end = data_num
                
                batch_files = file_names[start:end]
                batch_data = [caffe.io.load_image(os.path.join(data_dir, category_dir, f)) for f in batch_files]
                
                # predict
                predictions = self.net.predict(batch_data, oversample=False)
                
                # extruct features
                features = self.net.blobs[layer_name].data.reshape(n_inputs, n_dim)[0:end - start]
                # features = [self.net.blobs[layer_name].data.reshape(n_inputs, n_dim)[4]]
                # features = [self.net.blobs[layer_name].data[0].flatten().tolist()]
                all_features = np.vstack([all_features, features])
            self.logger.info("progress: %d / %d", end, data_num)

        # scaled_feature = preprocessing.scale(flatten_feature)
        self.logger.info("saving features to  %s", save_file)
        np.save(save_file, all_features)
        self.logger.info("saving category to  %s", cat_file)
        cat_arr = np.array(cat_list)
        np.save(cat_file, cat_arr)
        print cat_arr.shape
        print all_features.shape

    def create_libsvm_format(self, feature_file, category_file, save_file="all.txt", train_file="train.txt", test_file="test.txt", max_train_num=30, max_test_num=30):
        
        self.logger.info("loading features from %s and category from %s", feature_file, category_file)
        features = np.load(feature_file)
        cat_arr =  np.load(category_file)
        
        self.logger.info("saving libsvm features to %s", save_file)
        with open(save_file, 'w') as f:
            for i, (cat_num, feature) in enumerate(zip(cat_arr, features)):
                self.logger.info("progress: %d / %d", i + 1, len(cat_arr))
                f.write(str(cat_num) + " " + " ".join([ "%d:%s" % (j + 1, feat) for j, feat in enumerate(feature)]) + "\n")

        self.logger.info("max_train_num: %d max_test_num: %d", max_train_num, max_test_num)
        self.logger.info("saving libsvm features to train_file:%s & test_file: %s", train_file, test_file)
        with open(train_file, 'w') as f1:
            with open(test_file, 'w') as f2:
                cum = 0
                for k, g in itertools.groupby(cat_arr):
                    cat_len = len(list(g))
                    start = cum
                    end = start + cat_len
                    for i, (cat_num, feature) in enumerate(zip(cat_arr[start:end], features[start:end])):

                        if i < max_train_num:
                            self.logger.info("progress: %d / %d (train)", start + i + 1, len(cat_arr))
                            f1.write(str(cat_num) + " " + " ".join([ "%d:%s" % (j + 1, feat) for j, feat in enumerate(feature)]) + "\n")
                        elif max_train_num <= i and i < (max_train_num + max_test_num):
                            self.logger.info("progress: %d / %d (test)", start + i + 1, len(cat_arr))
                            f2.write(str(cat_num) + " " + " ".join([ "%d:%s" % (j + 1, feat) for j, feat in enumerate(feature)]) + "\n")
                        else:
                            self.logger.info("progress: %d - %d / %d (skip)", start + i + 1, end, len(cat_arr))
                            break
                    cum = end

    def train_by_lbsvm(self, feature_file, category_file, max_train_num=30, max_test_num=30):

        self.logger.info("loading features from %s and category from %s", feature_file, category_file)
        features = np.load(feature_file)
        cat_arr =  np.load(category_file)

        self.logger.info("creating train & test data")
        cum = 0
        train_feat = np.array([]).reshape(0, features.shape[1])
        test_feat = np.array([]).reshape(0, features.shape[1])
        train_cat = np.array([])
        test_cat = np.array([])
        for k, g in itertools.groupby(cat_arr):
            cat_len = len(list(g))
            train_start = cum
            train_end = train_start + max_train_num
            test_start = train_end
            test_end = train_start + cat_len
            if test_end > train_start + max_train_num + max_test_num:
                test_end = train_start + max_train_num + max_test_num

            train_feat = np.vstack([train_feat, features[train_start:train_end]])
            test_feat = np.vstack([test_feat, features[test_start:test_end]])
            train_cat = np.append(train_cat, cat_arr[train_start:train_end])
            test_cat = np.append(test_cat, cat_arr[test_start:test_end])
            cum = train_start + cat_len
        
        self.logger.info("train_feature is %s", train_feat.shape)
        self.logger.info("test_feature is %s", test_feat.shape)
        self.logger.info("train_cat is %s", train_cat.shape)
        self.logger.info("test_cat is %s", test_cat.shape)

        self.logger.info("training")
        clf = LinearSVC()
        clf.fit(train_feat, train_cat)
        self.logger.info("testing")
        print clf.score(test_feat, test_cat)

    def preprocess(self, file_path):
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', self.channel_swap)
        # mean pixel
        transformer.set_mean('data', self.mean)
        # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_raw_scale('data', self.raw_scale)
        # the reference model has channels in BGR order instead of RGB
        transformer.set_channel_swap('data', self.channel_swap)
        # Classify the image by reshaping the net for the single input then doing the forward pass.
        self.net.blobs['data'].reshape(1,3,227,227)
        self.net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(file_path))
        out = self.net.forward()
        print("Predicted class is #{}.".format(out['prob'].argmax()))
        
    def vis_square(self, data, padsize=1, padval=0):
        data -= data.min()
        data /= data.max()
        
        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
        
        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        
        plt.imshow(data)
        plt.show()
        
    def print_structure(self):
        for name, layer  in self.net.blobs.items():
            print "%s, %s" % (name, layer.data.shape)

    def print_params(self):

        # The parameters are net.params['name'][0].
        for name, params  in self.net.params.items():
            print "%s, %s" % (name, params[0].data.shape)

    def print_biases(self):

        # The biases are net.params['name'][1].
        for name, params  in self.net.params.items():
            print "%s, %s" % (name, params[1].data.shape)

    def plot_layer(self, layer_name):
        feat = self.net.blobs[layer_name].data[0]
        print feat.shape
        self.vis_square(feat, padval=1)
        # plt.imshow(feat[0])
        # plt.show()

    def draw_net(self, out_img_file):
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(self.model_file).read(), net)
        print('Drawing net to %s' % out_img_file)
        caffe.draw.draw_net_to_file(net, out_img_file)

    def fine_tuning(self, solver_file):
        pass

    def create_lmdb(self):
        pass

if __name__ == '__main__':

    logging.root.setLevel(level=logging.INFO)
    pass

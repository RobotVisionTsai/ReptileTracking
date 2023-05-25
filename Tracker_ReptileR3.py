import sys
import time
import numpy as np
import collections

import os
import argparse
import json
from PIL import Image
from matplotlib import pyplot as plt
# from scipy import io

import random
import math
# import logging
from hardnet import *
# logging.getLogger().setLevel(20)
# logging.basicConfig(level=logging.INFO, 
#                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     filename='/home/rvl224/ReptileLog.txt')
# logging.info('Reptile Tracker Imported')

# sys.path.insert(0,'/usr/lib/python2.7/dist-packages')
# import cv2

sys.path.insert(0,'D:/MyPython_Project/ReptileFPNTracker/modules')
# sys.path.insert(0,'../modules')
# sys.path.insert(0,'/home/rvl224/Docker/VOT2021/shared/ReptileFPNTracker/modules')
# sys.path.insert(0,'/workspace/shared/ReptileFPNTracker/modules')
from sample_generator import *
from data_prov import *
#from meta_model import *
from bbreg import *
from options import *

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from Tracker_Models import VGG_miniimagenet_r2
# from Tracker_Models import VGG_SDCNN1
# from Tracker_Models import VGG_SDCNN2

def extract_regions(image, samples, crop_size, padding):
    regions = np.zeros((len(samples),crop_size,crop_size,3),dtype='uint8')
    for i, sample in enumerate(samples):
        regions[i] = crop_image(image, sample, crop_size, padding, True)
    
    regions = regions.transpose(0,3,1,2)
    regions = regions.astype('float32')/255.
    return regions
    
# model = model = HarDNet(arch=85)
model = VGG_miniimagenet_r2()
# model = VGG_SDCNN()
# model = VGG_SDCNN2()

model = model.cuda()
for param in model.features.parameters():
    param.requires_grad = False

# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default

optimizer = optim.Adam([
                       {"params":model.convs.parameters()},
                       {"params":model.fc.parameters()},
                       {"params":model.resize.parameters()}], lr=1e-2, betas=(0, 0.999))
# optimizer = optim.Adam([
#                        {"params":model.convs.parameters()},
#                        {"params":model.fc.parameters()},
#                        {"params":model.resize.parameters()}], lr=1e-1, betas=(0, 0.999))
criterion = nn.CrossEntropyLoss()

def totorch_x(x):
    return Variable(torch.Tensor(x)).cuda()

def totorch_y(x):
    return Variable(torch.LongTensor(x)).cuda()

def train_on_batch(x, y):
    x = totorch_x(x)
    y = totorch_y(y)
    optimizer.zero_grad()
    ypred = model(x)[0]
    loss = criterion(ypred, y)
    loss.backward()
    optimizer.step()
    
def output_loss(x, y):
    x = totorch_x(x)
    y = totorch_y(y)
    optimizer.zero_grad()
    ypred = model(x)[0]
    loss = criterion(ypred, y)
    return loss.data()

def _test_predictions(test_set):
    x = totorch_x(test_set)
    return model(x)[0]

def _test_predictions_features(test_set):
    x = totorch_x(test_set)
    return model(x)[1]
#     for i in range(10):
#         x = model.base[i](x)
#     return x

def gen_batch(pos, neg, batch_size, num_step):
    #pos = pos.numpy()
    #neg = neg.numpy()
    pos_len = pos.shape[0]
    neg_len = neg.shape[0]
    
    task = np.concatenate((pos, neg), axis=0)
    #task = (task+128.0)/255.0
    #print(task.shape)
    pos_labels = np.ones(pos_len)
    #print(pos_labels.shape)
    neg_labels = np.zeros(neg_len)
    #print(neg_labels.shape)
    labels = np.concatenate((pos_labels, neg_labels), axis=0)
    #print(labels.shape)
    
    # iters = math.floor(128/batch_size)
    num_iters = 0
    #print('iters', iters)
    #print('num_iters', num_iters)
    while True:
        if num_iters >= num_step:
            #print(num_iters)
            return 
        
        permute = np.random.permutation(labels.shape[0])
        
        yield task[permute[0:0+batch_size]], labels[permute[0:0+batch_size]]
        
        num_iters = num_iters+1
        #print('num_iters', num_iters)
        
# pretrained_model_dict = torch.load('/home/rvl224/Docker/VOT2021/shared/ReptileTracker/_models/saved_model_pre_large2.pth')
# pretrained_optim_dict = torch.load('/home/rvl224/Docker/VOT2021/shared/ReptileTracker/_models/saved_optim_pre_large2.pth')
# pretrained_model_dict = torch.load('/home/rvl224/Docker/VOT2021/shared/ReptileFPNTracker/_models/saved_model_pre_large9_StepSize0.4.pth')
# pretrained_optim_dict = torch.load('/home/rvl224/Docker/VOT2021/shared/ReptileFPNTracker/_models/saved_optim_pre_large9_StepSize0.4.pth')
# pretrained_model_dict = torch.load('/workspace/shared/ReptileFPNTracker/_models/saved_model_pre_large9_StepSize0.4.pth')
# pretrained_optim_dict = torch.load('/workspace/shared/ReptileFPNTracker/_models/saved_optim_pre_large9_StepSize0.4.pth')
pretrained_model_dict = torch.load('D:/MyPython_Project/ReptileFPNTracker/_models/saved_model_pre_large9_StepSize0.4.pth')
pretrained_optim_dict = torch.load('D:/MyPython_Project/ReptileFPNTracker/_models/saved_optim_pre_large9_StepSize0.4.pth')
# pretrained_model_dict = torch.load('/home/rvl224/Research_Reptile/my_code/tracking/_models/saved_model_pre_sdcnn02_StepSize0.4.pth')
# pretrained_optim_dict = torch.load('/home/rvl224/Research_Reptile/my_code/tracking/_models/saved_optim_pre_sdcnn02_StepSize0.4.pth')
# pretrained_model_dict = torch.load('/home/rvl224/Research_Reptile/my_code/tracking/_models/saved_model_pre_sdcnn02.2_StepSize0.4.pth')
# pretrained_optim_dict = torch.load('/home/rvl224/Research_Reptile/my_code/tracking/_models/saved_optim_pre_sdcnn02.2_StepSize0.4.pth')

model.load_state_dict(pretrained_model_dict, strict=False)
# optimizer.load_state_dict(pretrained_optim_dict, strict=False)
# np.random.seed(int(time.time()))

seedid=int(time.time())
# seedid = 1648389623  # 0.317
# seedid = 1648512081 # 0.380, 0.316 fixed
# seedid = 1648515905 # 0.299, 0.302 fixed
# seedid = 1648512082 # 0.381
# seedid = 1648512084 # 0.354
np.random.seed(seedid)
# print('seedid: ',seedid)

# f = open("/workspace/shared/seedid.txt", "w")
f = open("D:/MyPython_Project/ReptileFPNTracker/seedid.txt", "w")
f.write(str(seedid))
f.close()

class ReptileTracker(object):
    def __init__(self, image, region, enable_diaplay):
        # np.random.seed(seedid)
        # print('seedid: ',seedid)
        # logging.info('Reptile Tracker Init')
#         model.load_state_dict(pretrained_model_dict)
#         optimizer.load_state_dict(pretrained_optim_dict)
        self.np_image = np.asarray(image)                                                                                
        self.target_bbox = np.array(region)
        self.crop_size = opts['img_size']
        self.padding = opts['padding']
        self.result_bb = self.target_bbox
        self.sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
        self.pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
        self.neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)
        self.bInited = True
        # logging.info('Reptile Tracker Init: gen_samples')
        pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                                   self.target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
        # if pos_examples is None:
            # pos_examples = gen_samples(SampleGenerator('uniform', image.size, 0.1, 1.2),
                                #    self.target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])  
        if len(pos_examples) == 0:
            self.bInited = False
            return
                                      
        neg_examples = np.concatenate([
                        gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1), 
                                    self.target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init']),
                        gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                                    self.target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init'])])
        # logging.info(pos_examples)
        # logging.info(neg_examples)
        # logging.info('Reptile Tracker Init: extract_regions')
        pos_regions = extract_regions(self.np_image, pos_examples, self.crop_size, self.padding)
        neg_regions = extract_regions(self.np_image, neg_examples, self.crop_size, self.padding)
 
        self.batch_size  = 64
        # self.batch_size  = 32
        # self.batch_size  = opts['n_pos_update']+opts['n_neg_update']
        self.step_num = 3
        # self.step_num = 1
        # self.outerstepsize = 0.8
        self.outerstepsize = 1.0

        # logging.info('Reptile Tracker Init: train_on_batch')
        for t, y  in gen_batch(pos_regions, neg_regions, self.batch_size, self.step_num):
            train_on_batch(t, y)

        # logging.info('Reptile Tracker Init: bbreg_examples')
        bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                    self.target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])            
        # if bbreg_examples is None:
            # bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.1, 1.2, 1.1),
                                    # self.target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])            
        
        # logging.info(bbreg_examples)
        # logging.info('Reptile Tracker Init: bb_regions')
        bb_regions = extract_regions(self.np_image, bbreg_examples, self.crop_size, self.padding)
        # logging.info(bb_regions)
        # logging.info('Reptile Tracker Init: bb_features')
        bb_features = _test_predictions_features(bb_regions)

        # logging.info('Reptile Tracker Init: BBRegressor')
        #print(bb_features.data.view(-1,256*13*13))
        self.bbreg = BBRegressor(image.size)
        # self.bbreg.train(bb_features.data.view(-1,320*13*13), bbreg_examples, self.target_bbox)
        # self.bbreg.train(bb_features.data.view(-1,256*13*13), bbreg_examples, self.target_bbox)
        # self.bbreg.train(bb_features.data.view(-1,256*26*26), bbreg_examples, self.target_bbox)
        self.bbreg.train(bb_features.data.contiguous().view(-1,256*13*13), bbreg_examples, self.target_bbox)

        self.pos_regions_all = [pos_regions]
        self.neg_regions_all = [neg_regions]
        #pos_regions_all = []
        #neg_regions_all = []
        # logging.info('Reptile Tracker Init: _test_predictions')
        pos_regions_scores = _test_predictions(pos_regions)
        neg_regions_scores = _test_predictions(neg_regions)
        pos_regions_scores = pos_regions_scores.data.clone().cpu().numpy()
        #print(pos_regions_scores[0:5])
        neg_regions_scores = neg_regions_scores.data.clone().cpu().numpy()
        #print(neg_regions_scores[0:5])
        self.pos_regions_scores_all=[pos_regions_scores]
        self.neg_regions_scores_all=[neg_regions_scores]

        self.index = 0

        self.display = enable_diaplay
        # self.display = True
        # logging.info('Reptile Tracker Init: display_init')
        if self.display: 
            self.display_init(image, self.target_bbox, region)

#     def __del__(self):
#         if self.display:
#             plt.close(self.fig)

    def track(self, image, gt):
        # np.random.seed(seedid)
        # logging.info('Reptile Tracker Track')
        if self.bInited is False:
            # logging.info('Reptile Tracker Track: self.bInited is False')
            # x=0
            # return self.result_bb, totorch_x(x)
            return self.result_bb, 0
            
        self.np_image = np.asarray(image)
        samples = gen_samples(self.sample_generator, self.target_bbox, opts['n_samples'])
        sample_regions = extract_regions(self.np_image, samples, self.crop_size, self.padding)
        sample_scores = _test_predictions(sample_regions)
    
        top_scores, top_idx = sample_scores.data.clone()[:,1].topk(5)
        # top_neg_scores, top_neg_idx = sample_scores.data.clone()[:,0].topk(5)
        target_score = top_scores.mean()
        # top_neg_score = top_neg_scores.mean()
	    ## 20190628 modified
        top_idx = top_idx.cpu().numpy()
        self.target_bbox = samples[top_idx].mean(axis=0)
	    # target_bbox = samples[top_idx].mean(axis=0)
	    ##
        # print(target_score, top_neg_score)
        # print(top_neg_score)
        # opts['success_thr'] = 0.0
        success = target_score > opts['success_thr']
        # success = target_score > top_neg_score-1.5
        # print(success)
        # result[i] = target_bbox

        if success:
            self.sample_generator.set_trans_f(opts['trans_f'])
        else:
            self.sample_generator.set_trans_f(opts['trans_f_expand'])

        if success:
    	    ## 20190628 modified	
            bbreg_samples = samples[top_idx]
 	        # bbreg_samples = samples[top_idx.cpu().numpy()]
 	        ##
            bb_regions = extract_regions(self.np_image, bbreg_samples, self.crop_size, self.padding)
            bb_features = _test_predictions_features(bb_regions)
        
            # bbreg_feats = forward_samples(online_net, image, bbreg_samples, out_layer='features')
            # bbreg_samples = self.bbreg.predict(bb_features.data.view(-1,320*13*13), bbreg_samples)
            # bbreg_samples = self.bbreg.predict(bb_features.data.view(-1,256*13*13), bbreg_samples)
            bbreg_samples = self.bbreg.predict(bb_features.data.contiguous().view(-1,256*13*13), bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            self.target_bbox = self.result_bb
            bbreg_bbox = self.result_bb 

        self.result_bb = bbreg_bbox
        self.index = self.index + 1

        if success:
            # Draw pos/neg samples
            pos_examples = gen_samples(self.pos_generator, self.target_bbox, 
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            neg_examples = gen_samples(self.neg_generator, self.target_bbox, 
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])
            # Extract pos/neg features
            #pos_feats = forward_samples(online_net, image, pos_examples, out_layer='features')
            #neg_feats = forward_samples(online_net, image, neg_examples, out_layer='features')
            pos_regions = extract_regions(self.np_image, pos_examples, self.crop_size, self.padding)
            neg_regions = extract_regions(self.np_image, neg_examples, self.crop_size, self.padding)
            pos_regions_scores = _test_predictions(pos_regions)
            neg_regions_scores = _test_predictions(neg_regions)
            pos_regions_scores = pos_regions_scores.data.clone().cpu().numpy()
            neg_regions_scores = neg_regions_scores.data.clone().cpu().numpy()
        
            self.pos_regions_all.append(pos_regions)
            self.neg_regions_all.append(neg_regions)
            self.pos_regions_scores_all.append(pos_regions_scores)
            self.neg_regions_scores_all.append(neg_regions_scores)

            if len(self.pos_regions_all) > opts['n_frames_long']:
                del self.pos_regions_all[0]
                del self.pos_regions_scores_all[0]
            if len(self.neg_regions_all) > opts['n_frames_short']:
                del self.neg_regions_all[0]
                del self.neg_regions_scores_all[0]

        # Short term update
        if not success:
            # print("Short term update!")
            nframes = min(opts['n_frames_short'],len(self.pos_regions_all))
            pos_regions_short = self.pos_regions_all[-nframes:]
            pos_data = np.concatenate(pos_regions_short)
            #pos_regions_scores_short = pos_regions_scores_all[-nframes:]
            #pos_scores_data = np.concatenate(pos_regions_scores_short)

            #neg_feats_short = neg_feats_all
            neg_regions_short = self.neg_regions_all[-nframes:]
            neg_data = np.concatenate(neg_regions_short)

            neg_regions_scores_short = self.neg_regions_scores_all[-nframes:]
            neg_scores_data = np.concatenate(neg_regions_scores_short)
            hard_neg_index = np.argsort(neg_scores_data[:,1])[::-1]
            #print(hard_neg_index.shape)
            #print(hard_neg_index.shape)
            neg_data = neg_data[hard_neg_index[0:int(hard_neg_index.shape[0]/2)]]
            #print(neg_data.shape)

            #print(pos_data.shape, neg_data.shape)
            # self.batch_size  = 64
            # self.step_num = 3
            for t, y  in gen_batch(pos_data, neg_data, self.batch_size, self.step_num):
                train_on_batch(t, y)
        # Long term update
        elif self.index % opts['long_interval'] == 0:
            # print("++++++Long term update!+++++++")
            weights_before = deepcopy(model.state_dict())
            # outerstepsize0 = 0.1
            pos_data = np.concatenate(self.pos_regions_all)
            neg_data = np.concatenate(self.neg_regions_all)
            neg_scores_data = np.concatenate(self.neg_regions_scores_all)
            hard_neg_index = np.argsort(neg_scores_data[:,1])[::-1]
            #print(hard_neg_index.shape)
            neg_data = neg_data[hard_neg_index[0:int(hard_neg_index.shape[0]/2)]]
            #print(neg_data.shape)
            
            # self.batch_size  = 64
            # self.step_num = 5
            for t, y  in gen_batch(pos_data, neg_data, self.batch_size, self.step_num):
                train_on_batch(t, y)
            
            weights_after = model.state_dict()
            # self.outerstepsize = outerstepsize0
            model.load_state_dict({name : 
                     weights_before[name] + (weights_after[name] - weights_before[name]) * self.outerstepsize 
                     for name in weights_before})  

        if self.display:
            self.display_result(image, gt)

        # thresholds = np.arange(0,1.05,0.05)
        # n_frame = len(gt)
        # success = np.zeros(len(thresholds))
        # iou = np.zeros(n_frame)
        # for i in range(n_frame):
        #     iou[i] = overlap_ratio(gt[i], result_bb[i])
        # for i in range(len(thresholds)):
        #     success[i] = sum(iou > thresholds[i])/n_frame

        # # Save result
        # res = {}
        # res['results'] = []
        # res['results'].append({'res': result_bb.round().tolist(), 'type': 'rect', 'success': success.tolist(),
        #                         'len': len(result_bb), 'anno':gt.round().tolist()})
        
        # json.dump(res, open('../result9/otb/' + seq + '_' + tracker_name + '.json', 'w'), indent=2)
        # io.savemat('../result9/otb/' + seq +'_' + tracker_name + '.mat',res)

        # return vot.Rectangle(self.result_bb[0], self.result_bb[1], self.result_bb[2], self.result_bb[3]), target_score
        return self.result_bb, target_score.cpu().numpy()

    def display_init(self, image, result, region):
        dpi = 80.0
        dpi_a = 100
        figsize = (image.size[0]/dpi_a, image.size[1]/dpi_a)

        self.fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        self.fig.add_axes(ax)
        self.im = ax.imshow(image)

        if region is not None:
            self.gt_rect = plt.Rectangle(tuple(region[:2]),region[2],region[3], 
                linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(self.gt_rect)
        
        self.rect = plt.Rectangle(tuple(self.result_bb[:2]),self.result_bb[2],self.result_bb[3], 
                linewidth=3, edgecolor="#ff00ff", zorder=1, fill=False)
        ax.add_patch(self.rect)

        plt.pause(.01)
        plt.draw()

    def display_result(self, image, gt):
        self.im.set_data(image)
        if gt is not None:
            self.gt_rect.set_xy(gt[:2])
            self.gt_rect.set_width(gt[2])
            self.gt_rect.set_height(gt[3])
        self.rect.set_xy(self.result_bb[:2])
        self.rect.set_width(self.result_bb[2])
        self.rect.set_height(self.result_bb[3])
        plt.pause(.01)
        plt.draw()
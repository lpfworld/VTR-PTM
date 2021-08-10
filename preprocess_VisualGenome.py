#! -*- coding: utf-8 -*-
# 使用Resnet101提取区域特征
# 注：部分区域图片有问题，需要额外处理

import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from bert4keras.backend import keras, K


# 图像模型
preprocessing_image = keras.preprocessing.image
preprocess_input = keras.applications.resnet.preprocess_input
image_model = keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', pooling='avg')


def preprocess_data(files, train=True):
    """读取并整理COCO的数据,提取目标特征.
    [
     {'region_feature', keywords': str, 'caption': str},
     {'region_feature', keywords': str, 'caption': str},
     ...
    ]
    """

    for _, file in tqdm(enumerate(files)):
        res = []
        
        try:
            image_data = json.load(open(folder+file), encoding='utf-8')
        except UnicodeDecodeError:
            print(folder+file)
            continue
            
        image_id = file.replace('json', 'jpg')
        img_path = './data/VisualGenome/VG_100K/%s' % image_id
            
        if not os.path.exists(img_path):
            continue
            
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        for region in image_data.values():
            r = {}
            
            region_img = img[region['y']:region['y']+region['height'], region['x']:region['x']+region['width']]
            # 处理有问题的区域图片
            if region_img.any():
                x = np.expand_dims(region_img, axis=0)
                x = preprocess_input(x)
                region_feature = image_model.predict(x)
                r['region_feature'] = region_feature.tolist()[-1]
            else:
                continue
                    
            r['caption'] = region["phrase"]
            
            keywords = ''
            for ob in region['objects']:
                keywords += ob['name'] + ' '
                r['keywords'] = keywords
                  
            res.append(r)
        
        if train:
            np.save('./data/VisualGenome/train2016/'+file.replace('json', 'npy'), res)
        else:
            np.save('./data/VisualGenome/valid2016/'+file.replace('json', 'npy'), res)
            

folder = './data/VisualGenome/annotation/regionfiles/'
files = os.listdir(folder)

# 以8:2的方式将数据集划分为训练集和测试集
split_idx = int(len(files)*0.8)
train_data, valid_data = files[:split_idx], files[split_idx:]

preprocess_data(train_data, train=True)

preprocess_data(valid_data, train=False)
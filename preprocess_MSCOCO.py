#! -*- coding: utf-8 -*-
# 使用Resnet101提取目标特征

import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from bert4keras.backend import keras, K


coco_train_data_path = './data/MSCOCO/annotation/regionfiles/train2014/'
coco_val_data_path = './data/MSCOCO/annotation/regionfiles/val2014/'

save_train_file = './data/MSCOCO/annotation/features/train2014/'
save_val_file = './data/MSCOCO/annotation/features/val2014/'


# 图像模型
preprocessing_image = keras.preprocessing.image
preprocess_input = keras.applications.resnet.preprocess_input
image_model = keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', pooling='avg')


def generate_object_data(folder, train=True):
    """读取并整理COCO的数据,提取目标特征.
    [
     {'image_features': [2048]},
     {'key_words': str, 'caption': str},
     {'key_words': str, 'caption': str},
     ...
    ]
    """
    
    files = os.listdir(folder)

    for file in tqdm(files):
        res = []
            
        image_data = json.load(open(folder+file))
        image_id = file.replace('json', 'jpg')
        
        if train:
            img_path = 'data/coco2014/train2014/%s' % image_id
        else:
            img_path = 'data/coco2014/val2014/%s' % image_id
            
        if not os.path.exists(img_path):
            continue
            
        # 计算整张图的特征, keras使用RGB，注意只需要转换一次，切图的时候就不需要了
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        image_features = image_model.predict(x)
        res.append({'image_features':image_features.tolist()[-1]})
        
        for image in image_data.values():
            obj = {}
            obj['caption'] = image["phrase"]
            
            key_words = ''

            for ob in image['objects']:
                key_words += ob['name'] + ' '
                
                obj['key_words'] = key_words
            
            res.append(obj)
        
        if train:
            np.save('./data/MSCOCO/annotation/features/train2014/'+file.replace('json', 'npy'), res)
        else:
            # npy文件最小
            np.save('./data/MSCOCO/annotation/features/val2014/'+file.replace('json', 'npy'), res)
            

generate_object_data(coco_train_data_path, True)

generate_object_data(coco_val_data_path, False)
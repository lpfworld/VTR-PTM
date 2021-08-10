from __future__ import print_function

import os
import cv2
import copy
import json
import numpy as np
from tqdm import tqdm


from keras.layers import *
from keras.models import Model
from bert4keras.layers import Loss
from bert4keras.optimizers import Adam
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import sequence_padding, is_string
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder

from caption_eval.custom_caption_eval import calculate_metrics


# bert配置
config_path = 'bert-model/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'bert-model/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'bert-model/uncased_L-12_H-768_A-12/vocab.txt'

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

# 模型配置
maxlen = 64
batch_size = 16


def read_object_data(folder):
    """读取并整理COCO的数据,包括caption, object, attributes 和 relationships , 同时提取目标特征.
    单个数据如下:
    [
     {'image_features': [2048]},
     {'key_words': str, 'caption': str},
     {'key_words': str, 'caption': str},
     ...
    ]
    
    返回数据格式:
    -valid:
    [{'image_id':str,
      'features': [2048],
      'caption': [str, str, str, str, str],
      'objects_key_words': [str, str, str, str, str]},
    ...  
    ]
    """
    print('-Read data ...')
    res = []
    
    files = os.listdir(folder)

    # 读取valid的caption
    data = json.load(open('data/coco2014/annotations/captions_val2014.json'))
    images = {}
    for img in data['images']:
        images[img['id']] = {
            'image_id': img['file_name'],
            'caption': [],
        }
    for caption in data['annotations']:
        images[caption['image_id']]['caption'].append(caption['caption'])
    captions = {}
    for img in images.values():
        captions[img['image_id']] = img['caption']
            
    # 读取image features 和 关键字
    for _, file in tqdm(enumerate(files)):
        file_path = folder + file
        data = np.load(file_path, allow_pickle=True)
        
        image = {}
        image_id = file.replace('npy', 'jpg')
        image['image_id'] = image_id
        image['features'] = np.array(data[0]['image_features'])
        image['caption']  = captions[image_id]
        image['objects_key_words']  = []

        for d in data[1:]:
            image['objects_key_words'].append(d['key_words'])
            
        res.append(image)
    
    return res
                
valid_data = read_object_data('./data/MSCOCO/annotation/features/val2014/')


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss
    
# 条件全连接层
x_in = Input(shape=(2048,), name='image_features')
    
# Bert模型
model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    layer_norm_cond=x_in,
    layer_norm_cond_hidden_size=512,
    layer_norm_cond_hidden_act='swish',
    additional_input_layers=x_in,
)

output = CrossEntropy(2)(model.inputs[0:2] + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoCaption(AutoRegressiveDecoder):
    """img2seq解码器
    """
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids, image = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids, image])[:, -1]

    def generate(self, inputs, features, topk=1):
        token_ids, segment_ids = tokenizer.encode(inputs, max_length=maxlen)
        output_ids = self.beam_search([token_ids, segment_ids, features], topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autocaption = AutoCaption(
    start_id=None,
    end_id=tokenizer._token_end_id,
    maxlen=maxlen
)


# 选取5张图片测试结果
eval_samples_id = ['COCO_val2014_000000549965.jpg', 'COCO_val2014_000000355776.jpg', 'COCO_val2014_000000539551.jpg', 'COCO_val2014_000000009003.jpg', 'COCO_val2014_000000447242.jpg']
eval_samples = []

for D in valid_data:
    if D['image_id'] in eval_samples_id:
        eval_samples.append(D)

def save_eval(epoch):
    res = {}
    for sample in eval_samples:
        res[sample['image_id']] = []
        for keyword in sample['objects_key_words']:
            pred = {}
            pred['keyword'] = keyword
            pred['caption'] = autocaption.generate(keyword, sample['features'])
            res[sample['image_id']].append(pred)
    
    with open('models/coco2014/base_kw/eval_samples_{}.json'.format(epoch), "w") as f:
        json.dump(res, f)
        
        
def caption_eval(epoch):
    save_eval(epoch)
        
    datasetGTS = {}
    datasetRES = {}
        
    GTS_annotations = []
    RES_annotations = []
    
    samples = [valid_data[i] for i in np.random.choice(len(valid_data), 5000, replace=False)]
    
    imgIds = 0
    for _, sample in tqdm(enumerate(samples), desc='Reading data'):
        for inputs in sample['objects_key_words']:
            res = {}
            res[u'image_id'] = imgIds
            res[u'caption'] = autocaption.generate(inputs, sample['features'])
            RES_annotations.append(res)
            
            for caption in sample['caption']:
                gts = {}
                gts[u'image_id'] = imgIds
                gts[u'caption'] = caption
                GTS_annotations.append(gts)
            
            imgIds += 1
            
    imgIds = range(imgIds)
        
    datasetGTS['annotations'] = GTS_annotations
    datasetRES['annotations'] = RES_annotations
    
    print(u'-Calculating scores ...')
    scores = calculate_metrics(imgIds, datasetGTS, datasetRES)
    print(scores)
    
    scores['epoch'] = epoch
    
    save_path = 'models/coco2014/base_kw/'
    
    with open(save_path + 'caption_eval.txt', "a") as f:
        f.write(str(scores) + '\n')
        

for epoch in range(10):
    model.load_weights('models/coco2014/base_kw/3/model_{}.weights'.format(epoch))
    caption_eval(epoch)
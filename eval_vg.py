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


def read_data(folder):
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
            
    # 读取image features 和 关键字
    for _, file in tqdm(enumerate(files)):
        file_path = folder + file
        data = np.load(file_path, allow_pickle=True)
        
        image = {}
        image_id = file.replace('npy', 'jpg')
        image['image_id'] = image_id
        image['regions']  = []

        for region in data:
            region['region_feature'] = np.array(region['region_feature'])
            image['regions'].append(region)
            
        res.append(image)
    
    return res 
    
valid_data = read_data('./data/VisualGenome/valid2016/')
print('Valid data: ', len(valid_data))


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


def just_show():
    samples = [valid_data[i] for i in np.random.choice(len(valid_data), 2, replace=False)]
    for img in samples:
        region = np.random.choice(img['regions'])
        print(u'image_id:', img['image_id'])
        print(u'key_words:', region['keywords'])
        print(u'predict:', autocaption.generate(region['keywords'], region['region_feature']))
        print(u'references:', region['caption'])
        print()
        
        
def caption_eval(epoch):
    just_show()
        
    datasetGTS = {}
    datasetRES = {}
        
    GTS_annotations = []
    RES_annotations = []
    
    samples = [valid_data[i] for i in np.random.choice(len(valid_data), 2500, replace=False)]
    
    imgId = 0
    for _, sample in tqdm(enumerate(samples), desc='Reading data'):
        for region in sample['regions']:
            res = {}
            res[u'image_id'] = imgId
            res[u'caption'] = autocaption.generate(region['keywords'], region['region_feature'])
            RES_annotations.append(res)
            
            gts = {}
            gts[u'image_id'] = imgId
            gts[u'caption'] = region['caption']
            GTS_annotations.append(gts)
            
            imgId += 1
            
    imgIds = range(imgId)
    datasetGTS['annotations'] = GTS_annotations
    datasetRES['annotations'] = RES_annotations
    
    print(u'-Calculating scores ...')
    scores = calculate_metrics(imgIds, datasetGTS, datasetRES)
    print(scores)
    
    scores['epoch'] = epoch
    
    save_path = 'models/VisualGenome/conditional_kw/'
    
    with open(save_path + 'caption_eval.txt', "a") as f:
        f.write(str(scores) + '\n')
        

for epoch in range(10):
    model.load_weights('models/VisualGenome/conditional_kw/model_{}.weights'.format(epoch))
    caption_eval(epoch)
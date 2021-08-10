#! -*- coding: utf-8 -*-
# unilm 做image caption任务，coco数据集
# 通过Conditional Layer Normalization融入条件信息


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
from optparse import OptionParser


parser = OptionParser()

parser.add_option("--with_pretrain_weights", dest="with_pretrain_weights", help="pretrain weights.", default=False)

(options, args) = parser.parse_args()


# 模型配置
maxlen = 64
batch_size = 16
steps_per_epoch =  20000
epochs = 10


# bert配置
config_path = 'bert-model/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'bert-model/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'bert-model/uncased_L-12_H-768_A-12/vocab.txt'

if options.with_pretrain_weights:
    pretrain_model_weights_path = 'models/coco2014/back-up.weights'
else:
    pretrain_model_weights_path = None


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def read_object_data(folder, valid=False):
    """读取并整理COCO的数据,包括caption, object, attributes 和 relationships , 同时提取目标特征.
    单个数据如下:
    [
     {'image_features': [2048]},
     {'key_words': str, 'caption': str},
     {'key_words': str, 'caption': str},
     ...
    ]
    
    返回数据格式:
    -train:
    [{'key_words': str,
      'caption': str},
    ...
    ]
    
    -valid:
    [{'image_id':str,
      'caption': [str, str, str, str, str],
      'objects_key_words': [str, str, str, str, str]},
    ...  
    ]
    """
    print('-Read data ...')
    res = []
    
    files = os.listdir(folder)
    
    if valid:
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
            image['caption']  = captions[image_id]
            image['objects_key_words']  = []
            
            for d in data[1:]:
                image['objects_key_words'].append(d['key_words'])
            
            res.append(image)
    else:
        for _, file in tqdm(enumerate(files)):
            file_path = folder + file
            data = np.load(file_path, allow_pickle=True)
        
            for d in data[1:]:
                res.append(d)
        
    return res


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, D in self.sample(random):

            caption = D['caption']
            inputs = D['key_words']

            token_ids, segment_ids = tokenizer.encode(
                inputs, caption, max_length=maxlen
            )
            
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
                
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []

                
# 加载数据
train_data = read_object_data(
    './data/MSCOCO/annotation/features/train2014/', False
)
valid_data = read_object_data(
    './data/MSCOCO/annotation/features/val2014/', True
)
print('-Train data numbers: ', len(train_data))
print('-Valid data numbers: ', len(valid_data))


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


# Bert模型
model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens  # 只保留keep_tokens中的字，精简原字表
)


output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoCaption(AutoRegressiveDecoder):
    """img2seq解码器
    """
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, inputs, topk=1):
        token_ids, segment_ids = tokenizer.encode(inputs, max_length=maxlen)
        output_ids = self.beam_search([token_ids, segment_ids], topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autocaption = AutoCaption(
    start_id=None,
    end_id=tokenizer._token_end_id,
    maxlen=maxlen
)

        
def just_show():
    samples = [valid_data[i] for i in np.random.choice(len(valid_data), 2, replace=False)]
    for D in samples:
        inputs = np.random.choice(D['objects_key_words'])
        print(u'image_id:', D['image_id'])
        print(u'key_words:', inputs)
        print(u'predict:', autocaption.generate(inputs))
        print(u'references:', D['caption'])
        print()


def caption_eval(epoch, loss):
    datasetGTS = {}
    datasetRES = {}
        
    GTS_annotations = []
    RES_annotations = []
    
    imgIds = range(5000)
    samples = [valid_data[i] for i in np.random.choice(len(valid_data), 5000, replace=False)]
    
    for imgId, sample in tqdm(enumerate(samples), desc='Reading data'):
        
        res = {}
        res[u'image_id'] = imgId
        inputs = np.random.choice(sample['objects_key_words'])
        res[u'caption'] = autocaption.generate(inputs)
        RES_annotations.append(res)
            
        for caption in sample['caption']:
            gts = {}
            gts[u'image_id'] = imgId
            gts[u'caption'] = caption
            GTS_annotations.append(gts)
        
    datasetGTS['annotations'] = GTS_annotations
    datasetRES['annotations'] = RES_annotations
    
    print(u'-Calculating scores ...')
    scores = calculate_metrics(imgIds, datasetGTS, datasetRES)
    print(scores)
    
    scores['epoch'] = epoch
    scores['loss']  = loss
    
    save_path = 'models/coco2014/kw/'
    
    with open(save_path + 'caption_eval.txt', "a") as f:
        f.write(str(scores) + '\n')
    model.save_weights(save_path + 'model_{}.weights'.format(epoch))

    
class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存模型
        model.save_weights('models/coco2014/back-up.weights')
        
        # 演示效果
        just_show()
        
        # 保存模型
        caption_eval(epoch, logs['loss'])


if __name__ == '__main__':

    evaluator = Evaluate()
    train_generator = data_generator(train_data, batch_size)

    if pretrain_model_weights_path is not None and os.path.exists(pretrain_model_weights_path):
        print('-------------------load exists weights from: {}'.format(pretrain_model_weights_path))
        model.load_weights(pretrain_model_weights_path)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights(pretrain_model_weights_path)
    
"""
image_id: COCO_val2014_000000524611.jpg
key_words: train tracks
predict: a train that is sitting on the tracks.
references: [u'A train carrying chemical tanks traveling past a water tower.', u'Dual train tracks with a train on one of them and a water tower in the background.', u'a train some trees and a water tower ', u'Train on tracks with water tower for Davis Junction in the rear.', u'A train on a train track going through a bunch of trees.']
"""

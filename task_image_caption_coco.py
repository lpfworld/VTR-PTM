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

parser.add_option("--config_path", dest="config_path", help="BERT config path.", default='bert-model/uncased_L-12_H-768_A-12/bert_config.json')
parser.add_option("--checkpoint_path", dest="checkpoint_path", help="BERT checkpoint path.", default='bert-model/uncased_L-12_H-768_A-12/bert_model.ckpt')
parser.add_option("--dict_path", dest="dict_path", help="BERT dict_path path.", default='bert-model/uncased_L-12_H-768_A-12/vocab.txt')
parser.add_option("--with_pretrain_weights", dest="with_pretrain_weights", help="pretrain weights.", default=False)

parser.add_option("--maxlen", dest="maxlen", help="BERT的输入文本的最大长度.", default=64)
parser.add_option("--batch_size", dest="batch_size", help="一次训练所选取的样本数.", default=16)
parser.add_option("--steps_per_epoch", dest="steps_per_epoch", help="每个epoch需要训练的样本数,总样本数/batch_size得到.", default=20000)
parser.add_option("--epochs", dest="epochs", help="训练轮数.", default=10)


(options, args) = parser.parse_args()


# 模型配置
maxlen = options.maxlen
batch_size = options.batch_size
steps_per_epoch =  int(options.steps_per_epoch)
epochs = int(options.epochs)


# bert配置
config_path = options.config_path
checkpoint_path = options.checkpoint_path
dict_path = options.dict_path

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
      'features': [2048],
      'caption': str},
    ...
    ]
    
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
            image['features'] = np.array(data[0]['image_features'])
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
                d['features'] = np.array(data[0]['image_features'])
            
                res.append(d)
        
    return res


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_features, batch_token_ids, batch_segment_ids = [], [], []
        for is_end, D in self.sample(random):

            features = D['features']
            caption = D['caption']
            inputs = D['key_words']

            token_ids, segment_ids = tokenizer.encode(
                inputs, caption, max_length=maxlen
            )
            
            batch_features.append(features)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
                
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_features = np.array(batch_features)
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids, batch_features], None
                batch_features, batch_token_ids, batch_segment_ids = [], [], []

                
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
    for D in samples:
        features = D['features']
        inputs = np.random.choice(D['objects_key_words'])
        print(u'image_id:', D['image_id'])
        print(u'key_words:', inputs)
        print(u'predict:', autocaption.generate(inputs, features))
        print(u'references:', D['caption'])
        print()
        
        
def caption_eval(epoch, loss):
    datasetGTS = {}
    datasetRES = {}
        
    GTS_annotations = []
    RES_annotations = []
    
    samples = [valid_data[i] for i in np.random.choice(len(valid_data), 5000, replace=False)]
    
    imgId = 0
    for _, sample in tqdm(enumerate(samples), desc='Reading data'):
        for keyword in sample['objects_key_words']:
            res = {}
            res[u'image_id'] = imgId
            res[u'caption'] = autocaption.generate(keyword, sample['features'])
            RES_annotations.append(res)
            
            for caption in sample['caption']:
                gts = {}
                gts[u'image_id'] = imgId
                gts[u'caption'] = caption
                GTS_annotations.append(gts)
            
            imgId += 1
    
    imgIds = range(imgId)
    datasetGTS['annotations'] = GTS_annotations
    datasetRES['annotations'] = RES_annotations
    
    print(u'-Calculating scores ...')
    scores = calculate_metrics(imgIds, datasetGTS, datasetRES)
    print(scores)
    
    scores['epoch'] = epoch
    scores['loss']  = loss
    
    save_path = 'models/coco2014/base_kw/'
    
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

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

# 图像模型
preprocessing_image = keras.preprocessing.image
preprocess_input = keras.applications.resnet.preprocess_input
image_model = keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', pooling='avg')

# 模型配置
maxlen = 64
batch_size = 16


def read_qa_data(data='train2014'):
    """
    -Returns:
    [{'question': str, 'answer': str, 'image_feature': [2048]},
    ...
    ]
    """
    print('-Read {} data ...'.format(data))
    questions = json.load(open('data/VQA/v2_OpenEnded_mscoco_{}_questions.json'.format(data)), encoding='utf-8')
    annotations = json.load(open('data/VQA/v2_mscoco_{}_annotations.json'.format(data)), encoding='utf-8')
    img_files = os.listdir('data/MSCOCO/annotation/features/{}/'.format(data))
    all_img   = os.listdir('data/coco2014/{}/'.format(data))
    
    image_feature  = {}
    image_keywords = {}
    for _, img in tqdm(enumerate(img_files)):
        img_id = int(img.replace('.npy','').split('_')[-1])
        d = np.load('data/MSCOCO/annotation/features/{}/'.format(data)+img, allow_pickle=True)
        image_feature[img_id] = np.array(d[0]['image_features'])
        image_keywords[img_id] = ''
        for i in d[1:]:
            i = i["key_words"].strip().split(" ")
            for k in i:
                if k not in image_keywords[img_id]:
                    image_keywords[img_id] += ' ' + k
            
    answers = {}
    for annotation in annotations['annotations']:
        answers[annotation['question_id']] = annotation['multiple_choice_answer']
    
    qa = []
    for _, q in tqdm(enumerate(questions['questions'])):
        pair = {}
        pair['question_id'] = q['question_id']
        pair['question'] = q['question']
        pair['answer'] = answers[q['question_id']]
        
        if q['image_id'] in image_feature.keys():
            pair['image_feature'] = image_feature[q['image_id']]
            pair['keywords'] = image_keywords[q['image_id']]
        else:
            for img_id in all_img:
                if q['image_id'] == int(img_id.replace('.jpg','').split('_')[-1]):
                    img_path = 'data/coco2014/%s/%s' % (data, img_id)
                    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    x = np.expand_dims(img, axis=0)
                    x = preprocess_input(x)
                    f = image_model.predict(x)
                    if len(f) != 2048:
                        f = f[-1]
                    pair['image_feature'] = np.array(f)
                    pair['keywords'] = ''
                    
        # 找不到图片的用空白图片替代
        if not pair.__contains__('image_feature'):
            img = np.zeros((224,224,3))
            img.fill(225)
            x = np.expand_dims(img, axis=0)
            x = preprocess_input(x)
            f = image_model.predict(x)
            if len(f) != 2048:
                f = f[-1]
            pair['image_feature'] = np.array(f)
            pair['keywords'] = ''
        
        qa.append(pair)
            
    return qa

                
# 加载数据
valid_data = read_qa_data(data='val2014')
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

    def generate(self, keywords, question, features, topk=3):
        q_token_ids, q_segment_ids = tokenizer.encode(
            question, max_length=maxlen
        )
            
        k_token_ids, k_segment_ids = tokenizer.encode(
            keywords, max_length=maxlen
        )
            
        token_ids = k_token_ids + q_token_ids[1:]
        segment_ids = k_segment_ids + q_segment_ids[1:]
            
        output_ids = self.beam_search([token_ids, segment_ids, features], topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autocaption = AutoCaption(
    start_id=None,
    end_id=tokenizer._token_end_id,
    maxlen=maxlen
)


def vqa_predict():
    res = []
    for _, D in tqdm(enumerate(valid_data)):
        q = {}
        q['answer'] = autocaption.generate(D['keywords'], D['question'], D['image_feature'])
        q['question_id'] = D['question_id']
        res.append(q)
        
    with open('VQAEvaluation/Results/v2_OpenEnded_mscoco_val2014_real_results_{}.json'.format(epoch), 'w') as f:
        jsObj = json.dumps(res)  
        f.write(jsObj)

for epoch in range(10):
    model.load_weights('models/VQA/frcnn_qa/model_{}.weights'.format(epoch))
    vqa_predict()
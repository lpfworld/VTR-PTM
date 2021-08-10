#! -*- coding: utf-8 -*-
# bert做image caption任务，coco数据集
# 通过Conditional Layer Normalization融入条件信息
# 请参考：https://kexue.fm/archives/7124

from __future__ import print_function
import json
import os
import copy
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, is_string
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
import cv2

# 模型配置
maxlen = 64
batch_size = 16
steps_per_epoch = 26000
epochs = 10

# bert配置
config_path = 'bert-model/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'bert-model/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'bert-model/uncased_L-12_H-768_A-12/vocab.txt'
pretrain_model_weights_path = 'models/base/bert/model_base.weights'

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def read_train_caption(f):
    """读取并整理COCO的Caption数据
    """
    data = json.load(open(f))
    res = []

    images = {}
    for img in data['images']:
        images[img['id']] = {
            'image_id': img['file_name'],
            'caption': '',
            'url': img['coco_url']
        }
    for caption in data['annotations']:
        images[caption['image_id']]['caption'] = caption['caption']
        res.append(copy.deepcopy(images[caption['image_id']]))
        
    return res


def read_val_caption(f):
    """读取并整理COCO的Caption数据
    """
    data = json.load(open(f))
    images = {}
    for img in data['images']:
        images[img['id']] = {
            'image_id': img['file_name'],
            'caption': [],
            'url': img['coco_url']
        }
    for caption in data['annotations']:
        images[caption['image_id']]['caption'].append(caption['caption'])
    return list(images.values())


def read_image(f):
    """单图读取函数（对非方形的图片进行白色填充，使其变为方形）
    """
    img = cv2.imread(f)
    height, width = img.shape[:2]
    if height > width:
        height, width = img_size, width * img_size // height
        img = cv2.resize(img, (width, height))
        delta = (height - width) // 2
        img = cv2.copyMakeBorder(
            img,
            top=0,
            bottom=0,
            left=delta,
            right=height - width - delta,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    else:
        height, width = height * img_size // width, img_size
        img = cv2.resize(img, (width, height))
        delta = (width - height) // 2
        img = cv2.copyMakeBorder(
            img,
            top=delta,
            bottom=width - height - delta,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
    img = img.astype('float32')
    return img[..., ::-1]  # cv2的读取模式为BGR，但keras的模型要求为RGB


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_images, batch_token_ids, batch_segment_ids = [], [], []
        for is_end, D in self.sample(random):
            img = 'data/coco2014/train2014/%s' % D['image_id']
            # caption = np.random.choice(D['caption'])
            
            caption = D['caption']
            
            token_ids, segment_ids = tokenizer.encode(
                caption, max_length=maxlen
            )
            
            batch_images.append(read_image(img))
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_images = np.array(batch_images)
                batch_images = preprocess_input(batch_images)
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids, batch_images], None
                batch_images, batch_token_ids, batch_segment_ids = [], [], []


# 加载数据
train_data = read_train_caption(
    'data/coco2014/annotations/captions_train2014.json'
)
valid_data = read_val_caption(
    'data/coco2014/annotations/captions_val2014.json'
)


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉padding部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        if mask[1] is None:
            y_mask = 1.0
        else:
            y_mask = K.cast(mask[1], K.floatx())[:, 1:]
        y_true = y_true[:, 1:]  # 目标token_ids
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# 图像模型
MobileNetV2 = keras.applications.mobilenet_v2.MobileNetV2
preprocess_input = keras.applications.mobilenet_v2.preprocess_input
image_model = MobileNetV2(include_top=False, pooling='avg')
img_size = 299

# Bert模型
model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='lm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    layer_norm_cond=image_model.output,
    layer_norm_cond_hidden_size=128,
    layer_norm_cond_hidden_act='swish',
    additional_input_layers=image_model.input,
)

output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoCaption(AutoRegressiveDecoder):
    """img2seq解码器
    """
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        # inputs = [image] 图片的特征，每个step都是一样的
        # output_ids 是将每次的预测结果,将其当作下一次的输入，进行下一次的预测
        image = inputs[0]
        token_ids = output_ids
        segment_ids = np.zeros_like(token_ids)
        return model.predict([token_ids, segment_ids, image])[:, -1]

    def generate(self, image, topk=1):
        if is_string(image):
            image = read_image(image)
        image = preprocess_input(image)
        output_ids = self.beam_search([image], topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autocaption = AutoCaption(
    start_id=tokenizer._token_start_id,
    end_id=tokenizer._token_end_id,
    maxlen=maxlen
)


# ----------cation eval -----------
from caption_eval.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from caption_eval.pycocoevalcap.bleu.bleu import Bleu
from caption_eval.pycocoevalcap.meteor.meteor import Meteor
from caption_eval.pycocoevalcap.rouge.rouge import Rouge
from caption_eval.pycocoevalcap.cider.cider import Cider
from caption_eval.pycocoevalcap.spice.spice import Spice
from caption_eval.pycocoevalcap.wmd.wmd import WMD


class COCOEvalCap:
    def __init__(self, images, gts, res):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.params = {'image_id': images}
        self.gts = gts
        self.res = res

    def evaluate(self):
        imgIds = self.params['image_id']
        gts = self.gts
        res = self.res

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
            (WMD(),   "WMD"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                print("%s: %0.3f" % (method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


def calculate_metrics(rng, datasetGTS, datasetRES):
    imgIds = rng
    gts = {}
    res = {}

    imgToAnnsGTS = {ann['image_id']: [] for ann in datasetGTS['annotations']}
    for ann in datasetGTS['annotations']:
        imgToAnnsGTS[ann['image_id']] += [ann]

    imgToAnnsRES = {ann['image_id']: [] for ann in datasetRES['annotations']}
    for ann in datasetRES['annotations']:
        imgToAnnsRES[ann['image_id']] += [ann]

    for imgId in imgIds:
        gts[imgId] = imgToAnnsGTS[imgId]
        res[imgId] = imgToAnnsRES[imgId]

    evalObj = COCOEvalCap(imgIds, gts, res)
    evalObj.evaluate()
    return evalObj.eval
# ----------cation eval -----------


def just_show():
    samples = [valid_data[i] for i in np.random.choice(len(valid_data), 2)]
    for D in samples:
        img = 'data/coco2014/val2014/%s' % D['image_id']
        print(u'image_id:', D['image_id'])
        print(u'url:', D['url'])
        print(u'predict:', autocaption.generate(img))
        print(u'references:', D['caption'])
        print()
    
        
def caption_eval(epoch, loss):
    imgIds = range(5000)
        
    datasetGTS = {}
    datasetRES = {}
        
    GTS_annotations = []
    RES_annotations = []
    
    print(u'------------------- caption eval -----------------------------')
    samples = [valid_data[i] for i in np.random.choice(len(valid_data), 5000)]
    for idx, D in enumerate(samples):
        for caption in D['caption']:
            gts = {}
            gts[u'image_id'] = idx
            gts[u'caption'] = caption
            GTS_annotations.append(gts)
    
        res = {}
        res[u'image_id'] = idx
        img = 'data/coco2014/val2014/%s' % D['image_id']
        res[u'caption'] = autocaption.generate(img)
        RES_annotations.append(res)
        
    datasetGTS['annotations'] = GTS_annotations
    datasetRES['annotations'] = RES_annotations
    
    print(u'-------------------- calculating scores ------------------------------\n')
    scores = calculate_metrics(imgIds, datasetGTS, datasetRES)
    print(scores)
    
    scores['epoch'] = epoch
    scores['loss']  = loss
    with open("models/base/bert/base_unilm_caption_eval.txt", "a") as f:
        f.write(str(scores) + '\n')
    model.save_weights('models/base/bert/base_unilm_eval_{}.weights'.format(epoch))


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
#         if logs['loss'] <= self.lowest:
#             self.lowest = logs['loss']
#             model.save_weights('model/base/bert/model_base.weights')

#         # 保存loss
#         result = {}
#         result['epoch'] = epoch 
#         result['loss']  = logs['loss']
#         with open("model/base/bert/model_base_loss.txt", "a") as f:
#             f.write(str(result) + '\n')
            
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

    model.load_weights('models/base/bert/model_base.weights')
"""
image_id: COCO_val2014_000000524611.jpg
url: http://images.cocodataset.org/val2014/COCO_val2014_000000524611.jpg
predict: a train that is sitting on the tracks.
references: [u'A train carrying chemical tanks traveling past a water tower.', u'Dual train tracks with a train on one of them and a water tower in the background.', u'a train some trees and a water tower ', u'Train on tracks with water tower for Davis Junction in the rear.', u'A train on a train track going through a bunch of trees.']

image_id: COCO_val2014_000000202923.jpg
url: http://images.cocodataset.org/val2014/COCO_val2014_000000202923.jpg
predict: a baseball game in progress with the batter up to plate.
references: [u'Batter, catcher, and umpire anticipating the next pitch.', u'A baseball player holding a baseball bat in the game.', u'A baseball player stands ready at the plate.', u'Baseball players on the field ready for the pitch.', u'A view from behind a mesh fence of a baseball game.']
"""

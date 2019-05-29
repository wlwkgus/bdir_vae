import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

from PIL import Image

chars = ['0x21',
'0x2b',
'0x3141',
'0x314f',
'0x3159',
'0x3163',
'0x3a',
'0x44',
'0x4e',
'0x58',
'0x62',
'0x6c',
'0x76',
'0x22',
'0x2c',
'0x3142',
'0x3150',
'0x315a',
'0x31',
'0x3b',
'0x45',
'0x4f',
'0x59',
'0x63',
'0x6d',
'0x77',
'0x23',
'0x2d',
'0x3145',
'0x3151',
'0x315b',
'0x32',
'0x3c',
'0x46',
'0x50',
'0x5a',
'0x64',
'0x6e',
'0x78',
'0x24',
'0x2e',
'0x3147',
'0x3152',
'0x315c',
'0x33',
'0x3d',
'0x47',
'0x51',
'0x5b',
'0x65',
'0x6f',
'0x79',
'0x25',
'0x2f',
'0x3148',
'0x3153',
'0x315d',
'0x34',
'0x3e',
'0x48',
'0x52',
'0x5c',
'0x66',
'0x70',
'0x7a',
'0x26',
'0x30',
'0x314a',
'0x3154',
'0x315e',
'0x35',
'0x3f',
'0x49',
'0x53',
'0x5d',
'0x67',
'0x71',
'0x7b',
'0x27',
'0x3131',
'0x314b',
'0x3155',
'0x315f',
'0x36',
'0x40',
'0x4a',
'0x54',
'0x5e',
'0x68',
'0x72',
'0x7c',
'0x28',
'0x3134',
'0x314c',
'0x3156',
'0x3160',
'0x37',
'0x41',
'0x4b',
'0x55',
'0x5f',
'0x69',
'0x73',
'0x7d',
'0x29',
'0x3137',
'0x314d',
'0x3157',
'0x3161',
'0x38',
'0x42',
'0x4c',
'0x56',
'0x60',
'0x6a',
'0x74',
'0x7e',
'0x2a',
'0x3139',
'0x314e',
'0x3158',
'0x3162',
'0x39',
'0x43',
'0x4d',
'0x57',
'0x61',
'0x6b',
'0x75',]

# Data Loader for pytorch
from torch.utils.data import Dataset
from PIL import Image
from os import listdir
from tqdm import tqdm
import os
import random
import json

np.random.seed(403)


class FontDataset(Dataset):
    def __init__(self, dataset_path, train):
        self.chars = chars
        #         self.fonts = ['NanumMyongjo', 'Gaegu', 'Noto', 'NanumPen']
        self.fonts = ['NanumMyongjo', 'NanumPen', 'HiMelody', 'Noto', 'Gaegu', 'SongMyung', 'NanumBrush']
        self.base_font = 'NanumMyongjo'
        self.transform_path = 'trans'
        self.transform_length = 6
        self.train = train
        self.dataset_path = dataset_path
        self.test_count = 5
        self.train_count = 500
        self.train_data = list()
        self.test_data = list()

        base_font = self.base_font
        for font in tqdm(self.fonts):
            if font == self.base_font:
                continue
            for char in self.chars:
                image_paths = sorted(listdir(os.path.join(self.dataset_path, font, char)))
                np.random.shuffle(image_paths)
                for image_path in image_paths[:self.train_count]:
                    base_im_path = os.path.join(self.dataset_path, base_font, char)
                    number = int(image_path.split('.')[0].split('_')[-1])
                    base_im_path = os.path.join(self.dataset_path, base_font, char)

                    d = dict()
                    d['char_label'] = self.chars.index(char)

                    d['font'] = np.zeros(len(self.fonts), dtype=np.float32)
                    d['font'][self.fonts.index(font)] = 1
                    d['char'] = np.zeros(len(self.chars), dtype=np.float32)
                    d['char'][self.chars.index(char)] = 1

                    if int(char, 16) < 0x41:
                        d['char_class'] = 0
                    elif 0x41 <= int(char, 16) < 0x61:
                        d['char_class'] = 1
                    elif 0x61 <= int(char, 16) < 0x3131:
                        d['char_class'] = 2
                    elif 0x3131 <= int(char, 16) < 0x314F:
                        d['char_class'] = 3
                    else:
                        d['char_class'] = 4

                    d['transform'] = self.get_transform_array(
                        os.path.join(self.dataset_path, self.transform_path, '{}.json'.format(number)))
                    d['image_path'] = os.path.join(self.dataset_path, font, char, image_path)
                    d['base_image_path'] = os.path.join(base_im_path, random.choice(listdir(base_im_path)))
                    # print(d['image_path'])
                    self.train_data.append(d)

                for image_path in image_paths[-self.test_count:]:
                    base_im_path = os.path.join(self.dataset_path, base_font, char)
                    number = int(image_path.split('.')[0].split('_')[-1])
                    base_im_path = os.path.join(self.dataset_path, base_font, char)

                    d = dict()
                    d['char_label'] = self.chars.index(char)

                    d['font'] = np.zeros(len(self.fonts), dtype=np.float32)
                    d['font'][self.fonts.index(font)] = 1
                    d['char'] = np.zeros(len(self.chars), dtype=np.float32)
                    d['char'][self.chars.index(char)] = 1

                    if int(char, 16) < 0x41:
                        d['char_class'] = 0
                    elif 0x41 <= int(char, 16) < 0x61:
                        d['char_class'] = 1
                    elif 0x61 <= int(char, 16) < 0x3131:
                        d['char_class'] = 2
                    elif 0x3131 <= int(char, 16) < 0x314F:
                        d['char_class'] = 3
                    else:
                        d['char_class'] = 4

                    d['transform'] = self.get_transform_array(
                        os.path.join(self.dataset_path, self.transform_path, '{}.json'.format(number)))
                    d['image_path'] = os.path.join(self.dataset_path, font, char, image_path)
                    d['base_image_path'] = os.path.join(base_im_path, random.choice(listdir(base_im_path)))
                    self.test_data.append(d)

    def get_transform_array(self, path):
        with open(path, 'r') as f:
            d = json.loads(f.read())
        r = np.zeros(6, dtype=np.float32)
        r[0] = d['scale_x'] - 1.0
        r[1] = d['scale_y'] - 1.0
        r[2] = d['translate_percent_x']
        r[3] = d['translate_percent_y']
        r[4] = d['rotate'] / 360
        r[5] = d['shear'] / 28

        return r

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            sample = self.train_data[idx].copy()
            im = np.asarray(Image.open(sample['image_path']))
            base_im = np.asarray(Image.open(sample['base_image_path']))
            sample['data'] = np.reshape(im[..., 3], (-1,)).astype(np.float32)
            sample['base_data'] = np.reshape(base_im[..., 3], (-1,)).astype(np.float32)
            if np.random.random() > 0.5:
                sample['data'] = (sample['data'] > np.random.randint(100, 130)).astype(np.float32) * 255
            if np.random.random() > 0.5:
                sample['data'] += (np.random.randn(784) * 50 + 50)
                sample['data'] = np.clip(sample['data'], 0, 255)
            sample.pop('image_path')
            sample.pop('base_image_path')
            im.close()
            base_im.close()
            # print(sample)
            return sample
        else:
            sample = self.test_data[idx].copy()
            im = np.asarray(Image.open(sample['image_path']))
            base_im = np.asarray(Image.open(sample['base_image_path']))
            sample['data'] = np.reshape(im[..., 3], (-1,)).astype(np.float32)
            sample['base_data'] = np.reshape(base_im[..., 3], (-1,)).astype(np.float32)
            sample.pop('image_path')
            sample.pop('base_image_path')
            im.close()
            base_im.close()
            return sample


if __name__ == '__main__':
    data_set = FontDataset('/home/tony/work/font_transform/new_data/data/', train=True)
    print('Start iterating')
    for data in data_set:
        pass

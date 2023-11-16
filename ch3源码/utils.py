"""
其他工具
"""
import os

import numpy as np
from scipy.io.wavfile import read
import torch
import torch.nn.functional as F


# 数据集metafile 解析
def meta_parse(meta_file_path, sep='|'):
    current_dir = os.path.dirname(__file__)
    current_dir = current_dir.replace("\\", "/")
    with open(meta_file_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        items = [line.strip().split(sep=sep) for line in lines]
        items = [(current_dir+p, lab) for p, lab in items]
    return items # [(wave_path, label), ()....]


# load 音频数据，并转换成tensor
def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


# 将tensor送到gpu中
def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


# mish激活函数
def mish(x):
    return x*torch.tanh(F.softplus(x))

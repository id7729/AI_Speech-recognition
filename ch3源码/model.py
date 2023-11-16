"""
模型定义
"""
import torch
from torch import nn
from utils import mish
from cfg_parse import cfg


class PreNet(nn.Module):
    """
    前置变换层，对mel特征变换
    """
    def __init__(self, cfg):
        super(PreNet, self).__init__()
        self.cfg = cfg
        self.layer_0 = nn.Linear(self.cfg['n_mel_channels'], self.cfg['pre_layer_hid_dim'])
        self.layer_1 = nn.Linear(self.cfg['pre_layer_hid_dim'], self.cfg['pre_layer_out_dim'])
        self.drop = nn.Dropout(self.cfg['pre_layer_drop']) # 0.3

    # mel input :[bn, n_mel_channels, frames]: [bn, 40, 1001]
    def forward(self, x): # x: [bn, n_mel_chs, n_frames]
        x = x.permute(0, 2, 1) # [bn, n_frames, n_mel_chs]
        x = self.layer_0(x)
        x = mish(x) # mish(2018), relu(x=0), prelue, leakyrelue, sigmoud,
        x = self.drop(x)
        return self.layer_1(x) # [bn, n_frames, trans_layer_out_dim]: [bn, 1001, 512]


class FCNet(nn.Module):
    """
    分类器：对最终的结果分类， 2层linear layer
    """
    def __init__(self, cfg):
        super(FCNet, self).__init__()
        self.cfg = cfg
        # 如果是双向的RNN最终的output的dimension要乘以2
        in_dim = self.cfg['rnn_hid_dim'] * (2 if self.cfg['bidirect'] else 1)

        self.layer_0 = nn.Linear(in_dim, self.cfg['fc_layer_dim'])
        self.layer_1 = nn.Linear(self.cfg['fc_layer_dim'], self.cfg['n_classes'])
        self.drop = nn.Dropout(self.cfg['fc_layer_drop'])

    def forward(self, x): # x: [bn, 1024/512]
        x = self.layer_0(x)
        x = mish(x)
        x = self.drop(x)
        return self.layer_1(x) # [bn, 6]


class ConvNorm(torch.nn.Module):
    """
    对conv1d的简单封装，主要是权重初始化
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Model(nn.Module):
    """
    模型构建
    """
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        # 前置网络
        self.prenet = PreNet(self.cfg)

        # 3层 conv1d 对转换后的特征进行短距离特征抽取
        conv_layers = []
        for _ in range(self.cfg['n_conv_layers']):
            conv_layer = nn.Sequential(
                ConvNorm(in_channels=self.cfg['conv_dim'],
                         out_channels=self.cfg['conv_dim'],
                         kernel_size=self.cfg['kernel_size'],
                         padding=int(self.cfg['kernel_size']//2)),
                nn.BatchNorm1d(self.cfg['conv_dim'])
            )
            conv_layers.append(conv_layer)
        self.convolution1ds = nn.ModuleList(conv_layers) # list
        self.convDrop = nn.Dropout(self.cfg['conv_drop'])

        # 用双向gru再次进行长距离特征抽取
        self.gru = nn.GRU(input_size=self.cfg['conv_dim'],
                          hidden_size=self.cfg['rnn_hid_dim'],
                          num_layers=self.cfg['rnn_layers'],
                          bidirectional=self.cfg['bidirect'])
        # 分类器
        self.fc = FCNet(self.cfg)

    def forward(self, x):
        x = self.prenet(x)  # [bn, n_frames, chs] : [bn, n_frames, chs]
        x = x.permute(0, 2, 1) # [bn, chs, n_frames]
        for conv in self.convolution1ds:
            x = self.convDrop(mish(conv(x))) # [bn, chs, n_frames]
        x = x.permute(2, 0, 1) # [n_frames, bn, chs]
        rnn_out, hn = self.gru(x) # [n_frames, bn, channel]: [1001, bn, 1024/512]
        out = self.fc(rnn_out[-1]) # [bn, 6]
        return out

if __name__ == '__main__':

    x = torch.rand(size=(3, 40, 1001))
    model = Model(cfg)
    x = model(x)
    print(x) # [3, 6]: log prob -> sotfmax- > [3, 6]

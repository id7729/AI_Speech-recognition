# Author: xixi
# Date: 2021.02.16

# Content:
# ## 1. mel 谱抽取的一般步骤
# ## 2.mel spectrum VS magnitude spectrum

import numpy as np
from librosa.filters import mel as librosa_mel_fn
import matplotlib.pyplot as plt


class FeatureExt(object):
    def __init__(self, sr, n_mel_channels):
        """
        初始化函数
        :param sr:对梅尔特征抽取的采样率
        :param n_mel_channels:
        """
        self.sr = sr
        self.frame_size = int(25*sr/1000) # 25ms(400采样点)的窗长
        self.frame_stride = int(10*sr/1000) # 10ms(160采样点)的窗移
        self.n_mel_channels = n_mel_channels # 需要的mel通道个数
        self.fmin, self.fmax = 0, int(sr/2)
        self.NFFT = 512
        self.mel_bias = librosa_mel_fn(sr, self.NFFT, self.n_mel_channels, self.fmin, self.fmax)

    def mel_calc(self, signal):

        # A 预先处理 (opt.)： 高频抬升， 系数一般选择 0.97 左右
        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        # B 分窗：语音中时域到时频域的方法
        signal_length = len(emphasized_signal)
        num_frames = int(np.floor(float(np.abs(signal_length -self.frame_size) ) /self.frame_stride)) + 1
        pad_signal_length = num_frames * self.frame_stride + self.frame_size
        z = np.zeros((pad_signal_length -signal_length))
        pad_signal = np.append(emphasized_signal, z)    # 分窗不够以后的填充
        indices = np.tile(np.arange(0, self.frame_size), (num_frames, 1)) + np.tile \
            (np.arange(0, num_frames * self.frame_stride, self.frame_stride), (self.frame_size, 1)).T
        frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]

        # C 窗函数(opt.)：抑制频谱泄漏，还有汉宁窗，矩形窗， povey窗
        frames *= np.hamming(self.frame_size)

        # D 短时傅里叶变换(stft)： 点数选择原则： 最小2^n大于窗长的点, 512 > 400
        frames_fft = np.fft.rfft(frames, self.NFFT) # stft

        # E 能量谱计算
        mag_frames = np.absolute(frames_fft)  # Magnitude of the FFT

        # F 三角mel滤波器组 apply
        mel = np.dot(self.mel_bias, mag_frames.T)
        return mel

    @staticmethod
    def display(mel):
        mel = np.flipud(mel)
        plt.imshow(mel)
        plt.show()


if __name__ == '__main__':
    from wav_proc import *
    wave_path = './data/16k-2bytes-mono.wav'
    WP = WaveProc()
    sr, duration, channel_num, data = WP.wave_read(wave_path)

    FE = FeatureExt(16000, 40) # 40, 80
    mel = FE.mel_calc(data)
    print(mel.shape)
    FE.display(mel)

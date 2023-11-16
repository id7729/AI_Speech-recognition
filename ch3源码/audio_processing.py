"""
音频处理相关
"""
import torch
from librosa.filters import mel as librosa_mel_fn
from stft import STFT


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """动态范围压缩
    :param x: 输入mel
    :param C: 压缩系数
    :param clip_val: 避免log0
    :return: 压缩变换后的特征
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


class MelSpec(torch.nn.Module):
    """
    这个类负责计算mel特征，并进行特征压缩
    """
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=40, sampling_rate=16000, mel_fmin=0.0,
                 mel_fmax=8000.0):
        """ mel 特征抽取
        :param filter_length: fft采样点数
        :param hop_length:  移动 stride
        :param win_length: 窗长
        :param n_mel_channels: mel channel 个数
        :param sampling_rate: 采样率
        :param mel_fmin:   最小截止频率
        :param mel_fmax:  最大截止频率
        """
        super(MelSpec, self).__init__()
        self.n_mel_channels = n_mel_channels    # 40
        self.sampling_rate = sampling_rate      # 16000
        self.stft_fn = STFT(filter_length=filter_length, hop_length=hop_length, win_length=win_length)
        mel_bias = librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_bias = torch.from_numpy(mel_bias).float()
        self.register_buffer('mel_bias', mel_bias)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """ mel 特征计算
        :param y: 幅值归一化后的音频数据
        :return: mel 特征
        """
        assert torch.min(y) >= -1 and torch.max(y) <= 1     # 归一化的判断
        magnitudes, phase = self.stft_fn.transform(y)   # 傅里叶变换
        magnitudes = magnitudes.data                    # 幅度谱
        mel_output = torch.matmul(self.mel_bias, magnitudes) # apply mel 三角滤波器组
        mel_output = self.spectral_normalize(mel_output) # 动态范围压缩 normalization
        return mel_output


if __name__ == '__main__':
    from utils import load_wav_to_torch
    wav = load_wav_to_torch('./dataset/cough/-HG6SJVD3mQ_0.000.wav')[0]
    mel_fn = MelSpec(filter_length=512,hop_length=160,win_length=400,n_mel_channels=40,sampling_rate=16000, mel_fmin=50,mel_fmax=800)
    wav_norm = wav / 32768.
    wav_norm = wav_norm.unsqueeze(0)
    mels = mel_fn.mel_spectrogram(wav_norm)
    print(mels)

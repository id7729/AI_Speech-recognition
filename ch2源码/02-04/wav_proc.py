# Author: xixi
# Date: 2021.02.16

# ## CONTENTS:
# #### 1. 了解一些音频的常用操作的库: wave, scipy.io.wavefile, librosa, waveio, sox ....ffmpeg...
# #### 2. 音频的读写
# #### 3. 三个域: time, frequency, time-freq

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf


class WaveProc(object):

    @staticmethod
    def wave_read(wave_path):
        """ 单声道 wav 文件打开
        :param wave_path: wav 文件路径
        :return: ndarray 数据和采样率...
        """
        assert wave_path.endswith('wav'), 'Not Supported File Format!'
        sr, wave_data = wf.read(wave_path) # [frame, channels]:[16000, 2]

        frame_num = wave_data.shape[0]
        duration = frame_num / sr

        if len(wave_data.shape) == 1:
            channel_num = 1
        else:
            channel_num = 2

        return sr, duration, channel_num, wave_data

    @staticmethod
    def pcm_read(pcm_path, sr):
        assert pcm_path.endswith('pcm'), 'Not Supported File Format!'

        wave_data = np.fromfile(pcm_path, dtype=np.short)

        frame_num = wave_data.shape[0]
        duration = frame_num / sr

        if len(wave_data.shape) == 1:
            channel_num = 1
        else:
            channel_num = 2
        return sr, duration, channel_num, wave_data

    def time_domain_display(self, wav_path):
        sr, duration, channel_num, data = self.wave_read(wav_path)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Time Domain')
        if channel_num == 1:
            plt.plot(data)
        elif channel_num == 2:
            plt.subplot(211)
            plt.plot(data[:, 0])
            plt.subplot(212)
            plt.plot(data[:, 1])
        plt.show()

    @staticmethod
    def wave_write(wave_data, output_path, sr):
        """ 写音频
        :param wave_data: ndarray 的音频数据
        :param output_path: 写路径
        :param sr: 采样率
        :return: None
        """
        assert output_path.endswith('wav'), "Not Supported File Format"

        wf.write(output_path, sr, wave_data)



# if __name__ == '__main__':
#     wave_path = './data/8k-2bytes-stereo.wav'
#     pcm_path = './data/8k-2bytes-mono.pcm'
# #
#     WP = WaveProc()
# #
#     sr, duration, channel_num, data = WP.wave_read(wave_path)
#     print("采样率：%d Hz" % sr)
#     print("长度: %d ms" % round(1000*duration))
#     print("通道数: %d ch" % channel_num)
# #
#     # sr, duration, channel_num, data = WP.pcm_read(pcm_path, 8000)
#     # print("采样率：%d Hz" % sr)
#     # print("长度: %d ms" % round(1000 * duration))
#     # print("通道数: %d ch" % channel_num)
# #
#     WP.time_domain_display(wave_path)
# #
#     amp_half_data = data*0.5
#
#     WP.wave_write(amp_half_data.astype(np.short), './data/half.wav', 16000, )


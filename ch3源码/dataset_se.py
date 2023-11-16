"""
数据的dataloader，作为原数据到模型之间的一个桥梁，负责数据的读取和特征的抽取
"""
import torch
from torch.utils.data import DataLoader
from utils import meta_parse, load_wav_to_torch
from audio_processing import MelSpec


class MelLoader(torch.utils.data.Dataset):

    def __init__(self, metafile_path, cfg):
        """ mel 特征 dataset 类
        :param metafile_path: 数据集metafile
        :param cfg: 配置文件
        """
        # meta parse
        self.items = meta_parse(metafile_path) # [(wav_path, label)]
        self.max_wav_value = cfg['max_wav_value'] # 32768.
        self.sampling_rate = cfg['sampling_rate']
        self.device = cfg['device']
        # mel 计算类实例
        self.mel_calc_inst = MelSpec(cfg['win_len'],        # stft win: 512
                                     cfg['hop'],            # stride: 160
                                     cfg['nfilter'],        # mel calc win: 400
                                     cfg['n_mel_channels'], # 40
                                     cfg['sampling_rate'],  # 16000
                                     cfg['mel_fmin'],       # 50Hz
                                     cfg['mel_fmax'])       # 800Hz

    def get_mel(self, wav_path):
        """
        :param wav_path: 音频路径
        :return: mel 特征
        """
        audio, sr = load_wav_to_torch(wav_path)
        assert sr == self.sampling_rate, 'sample rate not match!'
        audio_norm = audio / self.max_wav_value # 赋值归一化
        audio_norm = audio_norm.unsqueeze(0)
        melspec = self.mel_calc_inst.mel_spectrogram(audio_norm)    # mel计算
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, idx):
        wave_path, label = self.items[idx]
        return self.get_mel(wave_path).to(self.device), torch.tensor(int(label)).long().to(self.device)

    def __len__(self):
        return len(self.items)


# if __name__ == '__main__':
#     from cfg_parse import cfg
#
#     SE = MelLoader('./dataset/eval.csv', cfg)
#     eval_loader = DataLoader(SE, batch_size=13, shuffle=True, drop_last=True)
#
#     for batch in eval_loader:
#         mel, lab = batch
#         print(mel.size())
#         print(lab)
#         break

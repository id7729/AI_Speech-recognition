import time
import pyaudio
import torch
import scipy.io.wavfile
import numpy as np
from threading import Thread
from cfg_parse import cfg
import audio_processing as ap
from utils import *
from model import Model
import torch.nn.functional as F

FORMAT = pyaudio.paInt16  # 录制位宽：short
N_CHANNEL = 1   # 通道
SR = 16000  # 采样率
CHUCK_SIZE = int(10 * SR / 1000)  # 每次录制10ms
chuck_pos = 0 # 当前录制的位置
ring_length = 100 * 60 * 60  # 环形buffer的总长： 60 minutes
ring_buffer = [np.array([0]*CHUCK_SIZE)]*ring_length # 环形buffer
# 特征抽取实例
feat = ap.MelSpec(
            cfg['win_len'], cfg['hop'], cfg['nfilter'],
            cfg['n_mel_channels'], cfg['sampling_rate'], cfg['mel_fmin'],
            cfg['mel_fmax'])

# dataset/class-id.txt 读取
class_ids = {}
with open('./dataset/class-id.txt', 'r') as fr:
    for line in fr.readlines():
        line = line.strip()
        lab, cls = line.split('|')
        class_ids[int(cls)] = lab

# 模型构建
model = Model(cfg).to('cpu')
# load 模型
checkpoint = torch.load('./model_save/model_24_3000.pth', map_location=torch.device('cpu')) # 86.9%
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def recording(record_param):
    """ 实时音频录制
    :param record_param: 录制参数
    :return: None
    """
    global ring_buffer
    global ring_length
    global chuck_pos

    p = pyaudio.PyAudio()  # 实例化对象
    stream = p.open(format=record_param['FORMAT'],
                    channels=record_param['CHANNELS'],
                    rate=record_param['SR'],
                    input=True,
                    frames_per_buffer=record_param['CHUCK_SIZE'])  # 打开流，传入响应参数
    print('recording...')
    while 1:
        byte_stream = stream.read(record_param['CHUCK_SIZE'])
        data = np.frombuffer(byte_stream, np.short)
        ring_buffer[chuck_pos] = data
        chuck_pos += 1
        if chuck_pos == ring_length: # reached the buffer tail
            chuck_pos = 0
        # if chuck_pos == 100*5: # recording test
        #     scipy.io.wavfile.write('./tmp.wav', 16000, np.concatenate(ring_buffer[:chuck_pos]))
        #     break
    stream.stop_stream()  # 关闭流
    stream.close()
    p.terminate()


# 音频mel特征计算
def feature_calc(audio):
    assert audio.shape[0] == 16000 * 10, 'data length error'
    data = torch.FloatTensor(audio.astype(np.float32))
    data_norm = data / cfg['max_wav_value']
    data_norm = data_norm.unsqueeze(0)
    melspec = feat.mel_spectrogram(data_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec


def inference(mel):
    with torch.no_grad():
        cls = model(mel) # 前向推理
        log_prob = F.softmax(cls, dim=1).data.cpu().numpy()
        return log_prob


def predict(audio_from_mic):
    # mel calc
    mel = feature_calc(audio_from_mic)
    pred_lab = inference(mel.unsqueeze(0))
    return pred_lab


def streaming(callback, n_sec=10):
    global ring_buffer
    global chuck_pos
    time.sleep(11) # sleep 11 seconds
    print('streaming decoding running...')

    while 1:
        # 从ring_buffer里面取10s的音频
        pos = chuck_pos
        if pos - 100 * n_sec < 0:
            start_pos = 100*n_sec-pos
            pre = np.concatenate(ring_buffer[-start_pos:])
            flo = np.concatenate(ring_buffer[:pos])
            data = np.concatenate((pre, flo))
        else:
            data = np.concatenate(ring_buffer[pos - 100 * n_sec:pos])

        # 计算分类的结果
        ret = callback(data)
        prob = np.squeeze(ret)
        max_index = np.argsort(-prob).tolist()
        flush_log = ''
        for i, prob_idx in enumerate(max_index):
            if not i:
                flush_log += '\033[1;35m%s:%.2f\033[0m' % (class_ids[prob_idx], prob[prob_idx]) + '\n'
            else:
                flush_log += '%s:%.2f' % (class_ids[prob_idx], prob[prob_idx]) + '\n'
        print(flush_log)

        time.sleep(2)


if __name__ == '__main__':

    recording_args = {
        'ring_buffer': ring_buffer,
        'ring_length': ring_length,
        'FORMAT': FORMAT,
        'CHANNELS': N_CHANNEL,
        'SR': SR,
        'CHUCK_SIZE': CHUCK_SIZE
    }

    t_rec = Thread(target=recording, args=(recording_args, ))   # 麦克风声音的获取
    t_predict = Thread(target=streaming, args=(predict, 10))    # 对声音的解析
    t_rec.start()
    t_predict.start()


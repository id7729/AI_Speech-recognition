# Author: xixi
# Date: 2021.02.16

# ## CONTENTS:
# #### 1. libs and functions for opening/saving audio files
# #### 2. wave information check
# #### 3. three domains of wave feature

import os
import numpy as np
import matplotlib.pyplot as plt
import wave
import scipy.io.wavfile as wf

# ## libs for processing audio file ##
# wave | scipy.io.wavfile | waveio | librosa | ...

# # open an wave file
# wave_file_handle = wave.open('./data/16k-2bytes-mono.wav')
# param = wave_file_handle.getparams()    # get wave parameters
# print(param)
# nchannels, sample_width, sample_rate, num_frames = param[:4]
# wave_frame_bytes = wave_file_handle.readframes(num_frames)    # read as bytes
# wave_frame = np.fromstring(wave_frame_bytes,dtype=np.short)  # bytes to wave frames(short)
# print('Duration: %d ms' % int(1000*num_frames/sample_rate))

# # stereo
# wave_file_handle = wave.open('./data/8k-2bytes-stereo.wav')
# param = wave_file_handle.getparams()
# print(param)
# wave_frame_bytes = wave_file_handle.readframes(param[3])
# wave_frame = np.fromstring(wave_frame_bytes, dtype=np.short)
# channel_1 = wave_frame[:-1:2]
# channel_2 = wave_frame[1:-1:2]

# # open an pcm file:
# wave_frame = np.fromfile('./data/8k-2bytes-mono.pcm', dtype=np.short)

# # a simple method
# sample_rate, wave_frame = wf.read('./data/8k-2bytes-stereo.wav')
# print(wave_frame.shape)

# # save it
# wf.write('./data/8k-2bytes-stereo-save.wav', sample_rate, wave_frame)


# ## 3 types of domain: time domain, freq domain and time-freq domain

# open an wav file:
sample_rate, wave_frame = wf.read('./data/16k-2bytes-mono.wav')

# time domain plot

plt.xlabel('Time')
plt.ylabel('Amplitude')
duration = wave_frame.shape[0] / sample_rate
plt.title('Time Domain')
plt.plot(wave_frame)
plt.show()

# frequency domain plot
plt.xlabel('freq')
plt.ylabel('Power')
plt.title('Freq Domain')
fft_data = np.fft.fft(wave_frame)
# fft_data = np.abs(fft_data)
fft_data = fft_data.real
print(fft_data.shape)
x_dim = np.linspace(0, 8000,int(fft_data.shape[0]/2))
plt.plot(x_dim, fft_data[:int(fft_data.shape[0]/2)])
plt.show()

# tf-domain?

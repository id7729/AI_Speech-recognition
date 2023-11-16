from wav_proc import WaveProc
import matplotlib.pyplot as plt
import numpy as np

wave_path = './data/16k-2bytes-mono.wav'
WP = WaveProc()
sr, duration, channel_num, data = WP.wave_read(wave_path)


plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Time Domain')
plt.plot(data)
plt.show()

# frequency domain plot
plt.xlabel('freq')
plt.ylabel('Power')
plt.title('Freq Domain')
fft_data = np.fft.fft(data)
fft_data = np.abs(fft_data)
x_dim = np.linspace(0, 8000,int(fft_data.shape[0]/2))
plt.plot(x_dim, fft_data[:int(fft_data.shape[0]/2)])
plt.show()

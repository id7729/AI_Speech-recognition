"""
数据集的校验,确保送入模型中的数据是对的
"""
# import os
# import wave
#
# dataset_dir = './dataset/'
#
# # 需要校验的内容主要有：
# # nchannel:1(mono), width: 2bytes, sample_rate: 16000, nframes(采样点): 10*16000
# for root, sub_dir, items in os.walk(dataset_dir):
#     if not sub_dir:
#         for it in items:
#             f = wave.open(os.path.join(root, it), 'rb')
#             params = f.getparams()
#             nchannel, width, sample_rate, nframe = params[:4]
#
#             assert nchannel == 1 and width == 2 and sample_rate == 16000 and nframe == 10*16000
#         print(os.path.split(root)[1], 'passed!')
# print('dataset passed!')

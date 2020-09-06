#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/19 19:00
# @Author  : MengnanChen
# @File    : trim_silence.py
# @Software: PyCharm

import os
from tqdm import tqdm

import librosa
from scipy.io import wavfile
import numpy as np


def trim_wav_silence(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_wav_paths = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    for wav_path in tqdm(all_wav_paths):
        try:
            wav_basename = os.path.basename(wav_path)
            wav = librosa.load(wav_path, sr=16000)[0]
            wav = librosa.effects.trim(wav, top_db=33, frame_length=512, hop_length=128)[0]
            wav *= 32767 / max(0.01, np.max(np.abs(wav)))
            wavfile.write(os.path.join(output_dir, wav_basename), rate=16000, data=wav.astype(np.int16))
        except Exception as e:
            print('error: {} -> {}'.format(e, wav_path))


if __name__ == '__main__':
    input_dir = r'G:\workspace\thchs30\A8'
    output_dir = r'G:\workspace\thchs30\A8_trim'
    trim_wav_silence(input_dir, output_dir)

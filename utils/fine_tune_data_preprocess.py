#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/19 21:09
# @Author  : MengnanChen
# @File    : fine_tune_data_preprocess.py
# @Software: PyCharm

import os
from tqdm import tqdm


def thchs30_wavs_rename(input_dir):
    wav_paths = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    for wav_path in tqdm(wav_paths):
        wav_basename = os.path.basename(wav_path)
        trg_wav_basename = 'thchs30_{}'.format(wav_basename)
        os.rename(wav_path, os.path.join(input_dir, trg_wav_basename))


def thchs30_align_parrallel_corpus(wavs_input_dir, org_metadata_path, trg_metadata_path):
    wav_ids = [x[:-4] for x in os.listdir(wavs_input_dir)]
    output_lines = []
    with open(org_metadata_path, 'rb') as fin, open(trg_metadata_path, 'wb') as fout:
        for line in fin:
            line = line.decode('utf-8').strip('\r\n')
            wav_id, txt = line.split('|')
            if wav_id in wav_ids:
                output_lines.append((int(wav_id.split('_')[-1]), '{}|{}\n'.format(wav_id, txt)))
        output_lines = sorted(output_lines, key=lambda x: x[0])
        output_lines = [x[-1] for x in output_lines]
        for line in output_lines:
            fout.write(line.encode('utf-8'))


def split_index_text(input_path, index_output_path, text_output_path):
    with open(input_path, 'rb') as fin, open(index_output_path, 'wb') as fout_index, \
            open(text_output_path, 'wb') as fout_text:
        for line in fin:
            line = line.decode('utf-8').strip('\r\n ')
            if not line:
                continue
            index, text = line.split('|')
            fout_index.write('{}\n'.format(index).encode('utf-8'))
            fout_text.write('{}\n'.format(text).encode('utf-8'))


def merge_index_text(index_input_path, text_input_path, output_path):
    with open(index_input_path, 'rb') as fin_index, open(text_input_path, 'rb') as fin_text, \
            open(output_path, 'wb') as fout:
        indexes = []
        texts = []
        for line in fin_index:
            line = line.decode('utf-8').strip('\r\n ')
            if not line:
                continue
            indexes.append(line)
        for line in fin_text:
            line = line.decode('utf-8').strip('\r\n ')
            if not line:
                continue
            texts.append(line)
        assert len(indexes) == len(texts), 'length not equal'
        for line_index, line_text in zip(indexes, texts):
            fout.write('{}|{}\n'.format(line_index, line_text).encode('utf-8'))


if __name__ == '__main__':
    # wavs_input_dir = r'G:\workspace\thchs30\A8_trim2'
    # org_metadata_path = r'G:\workspace\thchs30\metadata'
    # trg_metadata_path = r'G:\workspace\thchs30\A8_trim2\A8_metadata'
    # thchs30_wavs_rename(wavs_input_dir)
    # thchs30_align_parrallel_corpus(wavs_input_dir, org_metadata_path, trg_metadata_path)

    # input_path = r'G:\workspace\thchs30\A8_trim2\A8_metadata'
    # index_output_path = r'G:\workspace\thchs30\A8_trim2\A8_metadata.index'
    # text_output_path = r'G:\workspace\thchs30\A8_trim2\A8_metadata.text'
    # split_index_text(input_path, index_output_path, text_output_path)

    index_input_path = r'G:\workspace\thchs30\A8_trim2\A8_metadata.index'
    text_input_path = r'D:\Git\Tacotron2CMC\utils\data\A8_metadata.pinyin'
    output_path = r'G:\workspace\thchs30\A8_trim2\A8_metadata.corpus'
    merge_index_text(index_input_path, text_input_path, output_path)

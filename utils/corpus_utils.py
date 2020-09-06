# -*- coding: utf-8 -*-
# @Time    : 2019/5/19 12:58
# @Author  : MengnanChen
# @FileName: corpus_utils.py
# @Software: PyCharm

import os
import glob
import shutil
from collections import Counter
import argparse
from tqdm import tqdm


def postprocess_corpus(input_dir, output_dir):
    input_txt_paths = glob.glob(os.path.join(input_dir, '*.txt')) + glob.glob(os.path.join(input_dir, '*.csv'))
    os.makedirs(output_dir, exist_ok=True)
    for input_txt_path in input_txt_paths:
        txt_basename = os.path.basename(input_txt_path)
        output_txt_path = os.path.join(output_dir, txt_basename)
        output_audio_dir = os.path.join(output_dir, 'wavs')
        os.makedirs(output_audio_dir, exist_ok=True)
        input_audio_dir = os.path.join(input_dir, 'wavs')
        with open(input_txt_path, 'rb') as fin, open(output_txt_path, 'wb') as fout:
            for line in tqdm(fin):
                line = line.decode('utf-8').strip('\r\n ')
                try:
                    index, text = line.split('|')
                    if not index:
                        print('error audio dir: {}, error line: {}'.format(input_audio_dir, line))
                        continue
                    text = text.strip('\r\n ')
                    if ('*' in index) or (not text):
                        continue
                    ori_audio_path = os.path.join(input_audio_dir, '{}.wav'.format(index))
                    trg_audio_path = os.path.join(output_audio_dir, '{}.wav'.format(index))
                    shutil.copy(ori_audio_path, trg_audio_path)
                    fout.write('{}|{}\n'.format(index, text).encode('utf-8'))
                except:
                    print('error audio dir: {}, except error line: {}'.format(input_audio_dir, line))


def check_file(input_dir):
    input_txt_paths = glob.glob(os.path.join(input_dir, '*.txt')) + glob.glob(os.path.join(input_dir, '*.csv'))
    input_audio_dir = os.path.join(input_dir, 'wav')
    for input_txt_path in input_txt_paths:
        text_indexes = []
        with open(input_txt_path, 'rb') as fin:
            for line in fin:
                line = line.decode('utf-8').strip('\r\n ')
                index = line.split('|')[0]
                text_indexes.append(index)

        text_index_counter = Counter(text_indexes)
        for k, v in text_index_counter.items():
            if v > 1:
                print('duplicate index: {}'.format(k))

        audio_names = os.listdir(input_audio_dir)
        audio_indexes = [os.path.basename(x)[:-4] for x in audio_names]
        text_indexes = set(text_indexes)
        audio_indexes = set(audio_indexes)
        print('length of text: {}'.format(len(text_indexes)))
        print('length of audio: {}'.format(len(audio_indexes)))

        if abs(len(text_indexes) - len(audio_indexes)) <= 0:
            print('no error')
        else:
            print('additional index of text lines: {}'.format(text_indexes - audio_indexes))
            print('additional index of audios: {}'.format(audio_indexes - text_indexes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='check', help='check/process ,check corpus after cleaning up')
    parser.add_argument('--input_dir', help='input dir of corpus(LJSpeech format)')
    parser.add_argument('--output_dir', help='output dir of neatening corpus, only need when task is process')
    args = parser.parse_args()

    if 'process' in args.task:
        postprocess_corpus(input_dir=args.input_dir, output_dir=args.output_dir)
    elif 'check' in args.task:
        check_file(input_dir=args.input_dir)
    else:
        raise Exception('not supported tasks')

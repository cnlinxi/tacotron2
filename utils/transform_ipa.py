# -*- coding: utf-8 -*-
# @Time    : 2019/5/6 10:06
# @Author  : MengnanChen
# @FileName: transform_ipa.py
# @Software: PyCharm

# https://github.com/dmort27/epitran

# cedict_1_0_ts_utf-8_mdbg.txt(cedict_1_0_ts_utf-8_mdbg.txt.gz) can be downloaded from
# https://www.mdbg.net/chinese/dictionary?page=cc-cedict

import os
import argparse
import jieba

dependency_dir = 'dependency'


def transform_chinese_ipa(input_path, cedict_file=os.path.join(dependency_dir, 'cedict_1_0_ts_utf-8_mdbg.txt')):
    try:
        import epitran
    except ImportError as e:
        print('failed to import epitran', e)

    output_corpus_path = os.path.join(os.path.dirname(input_path), 'result.ipa')
    epi_chi = epitran.Epitran('cmn-Hans', cedict_file=cedict_file)
    with open(input_path, 'rb') as fin, open(output_corpus_path, 'wb') as fout:
        for line in fin:
            line = line.decode('utf-8').strip('\r\n ')
            if len(line) <= 0:
                continue
            line = ' '.join(jieba.cut(line))
            ipa_txt = epi_chi.transliterate(line, normpunc=False, ligatures=False)
            fout.write('{}\n'.format(ipa_txt).encode('utf-8'))


def transform_english_ipa(input_path):
    try:
        import epitran
    except ImportError as e:
        print('failed to import epitran', e)

    output_corpus_path = os.path.join(os.path.dirname(input_path), 'result.ipa')
    epi_eng = epitran.Epitran('eng-Latn')
    with open(input_path, 'rb') as fin, open(output_corpus_path, 'wb') as fout:
        for line in fin:
            line = line.decode('utf-8').strip('\r\n ')
            if len(line) <= 0:
                continue
            line = epi_eng.transliterate(line, normpunc=False, ligatures=True)
            fout.write('{}\n'.format(line).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', required=True, default='chi', help='chi or eng')
    parser.add_argument('--input_path', required=True, help='input corpus, pure text')
    parser.add_argument('--cedict_file', help='cedict path when input corpus is chinese')
    args = parser.parse_args()

    if args.lang == 'chi':
        if args.cedict_file is not None:
            transform_chinese_ipa(input_path=args.input_path, cedict_file=args.cedict_file)
        else:
            transform_chinese_ipa(input_path=args.input_path)
    elif args.lang == 'eng':
        transform_english_ipa(input_path=args.input_path)
    else:
        raise Exception('Unsupported language')

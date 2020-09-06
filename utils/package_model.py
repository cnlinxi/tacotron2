#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/19 10:40
# @Author  : MengnanChen
# @File    : package_model.py
# @Software: PyCharm

import os
import zipfile

exclude_dirs = ['training_data', 'tacotron_output', 'LJSpeech-1.1']


def package_files(input_dir, output_path):
    zipin = zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED)
    for root, dirnames, filenames in os.walk(input_dir):
        if root in exclude_dirs:
            continue
        this_path = os.path.abspath('.')
        fpath = root.replace(this_path, '')
        for filename in filenames:
            zipin.write(os.path.join(root, filename), os.path.join(fpath, filename))
    zipin.close()


if __name__ == '__main__':
    input_dir = 'Tacotron2CMC'
    output_path = 'Tacotron2CMC_biaobei.zip'
    package_files(input_dir, output_path)

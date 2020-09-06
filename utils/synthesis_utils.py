# -*- coding: utf-8 -*-
# @Time    : 2019/2/12 11:08
# @Author  : MengnanChen
# @FileName: synthesis_utils.py
# @Software: PyCharm

'''
This script should be put the root of project
'''

import os
import time
import glob
import argparse
import zipfile
import shutil

import tensorflow as tf

from tacotron.synthesize import tacotron_synthesize
from synthesize import prepare_run, get_sentences


def zip_tacotron_wavs(args):
    zip_root_dir = 'wavs_zips'
    os.makedirs(zip_root_dir, exist_ok=True)
    wavs_dir = os.path.join('tacotron_output', 'logs-eval', 'wavs')
    exp_name = args.name
    ckpt_meta_path = os.path.join('logs-{}'.format(exp_name), 'taco_pretrained', 'checkpoint')
    with open(ckpt_meta_path, 'rb') as fin:
        line = fin.readline()
        line = line.decode('utf-8').strip('\r\n ')
        steps = line.split('-')[1].replace('"', '')
    zip_filename = 'wavs_{}_{}.zip'.format(exp_name, steps)
    z = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)  # 参数一：文件夹名
    for dirpath, dirnames, filenames in os.walk(wavs_dir):
        fpath = dirpath.replace(wavs_dir, '')  # 这一句很重要，不replace的话，就从根目录开始复制
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath + filename)
    z.close()
    shutil.move(zip_filename, os.path.join(zip_root_dir, zip_filename))


def main():
    accepted_modes = ['eval', 'synthesis', 'live']
    parser = argparse.ArgumentParser()
    ###
    parser.add_argument('--name', default='Tacotron-2',
                        help='Name of logging directory if the two models were trained together.')
    parser.add_argument('--text_list', default='valid_sentences.corpus',
                        help='Text file contains list of texts to be synthesized. Valid if mode=eval')
    parser.add_argument('--rm', default=True, help='remove tacotron output after zip file')
    parser.add_argument('--synthesis_mode', default='norm', help='sub_ckpt, all_ckpt or norm')
    parser.add_argument('--start_step', help='synthesis start step when synthesis_mode is sub_ckpt')
    parser.add_argument('--end_step', help='synthesis end step when synthesis_mode is sub_ckpt')
    parser.add_argument('--syn_interval', help='synthesis interval when synthesis_mode is sub_ckpt')
    ###
    parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--tacotron_name', help='Name of logging directory of Tacotron. If trained separately')
    parser.add_argument('--wavenet_name', help='Name of logging directory of WaveNet. If trained separately')
    parser.add_argument('--model', default='Tacotron')
    parser.add_argument('--input_dir', default='training_data/', help='folder to contain inputs sentences/targets')
    parser.add_argument('--mels_dir', default='tacotron_output/eval/',
                        help='folder to contain mels to synthesize audio from using the Wavenet')
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--mode', default='eval', help='mode of run: can be one of {}'.format(accepted_modes))
    parser.add_argument('--GTA', default='True',
                        help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
    parser.add_argument('--speaker_id', default=None,
                        help='Defines the speakers ids to use when running standalone Wavenet on a folder of mels. this variable must be a comma-separated list of ids')
    args = parser.parse_args()

    accepted_models = ['Tacotron', 'WaveNet', 'Tacotron-2']

    if args.model not in accepted_models:
        raise ValueError('please enter a valid model to synthesize with: {}'.format(accepted_models))

    if args.mode not in accepted_modes:
        raise ValueError('accepted modes are: {}, found {}'.format(accepted_modes, args.mode))

    if args.mode == 'live' and args.model == 'Wavenet':
        raise RuntimeError(
            'Wavenet vocoder cannot be tested live due to its slow generation. Live only works with Tacotron!')

    if args.GTA not in ('True', 'False'):
        raise ValueError('GTA option must be either True or False')

    taco_checkpoint, wave_checkpoint, hparams = prepare_run(args)
    sentences = get_sentences(args)

    assert args.synthesis_mode in ['norm', 'all_ckpt', 'sub_ckpt'], 'not supported synthesis mode'

    if args.synthesis_mode == 'norm':
        if args.model == 'Tacotron':
            _ = tacotron_synthesize(args, hparams, taco_checkpoint, sentences)
        else:
            raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))

        zip_tacotron_wavs(args)
        if args.rm:
            shutil.rmtree('tacotron_output')
    elif args.synthesis_mode == 'sub_ckpt':
        assert args.start_step.isnumeric(), 'start step must be number'
        assert args.end_step.isnumeric(), 'end step must be number'
        assert args.syn_interval.isnumeric(), 'synthesis interval must be number'
        exp_name = args.name
        ckpt_meta_path = os.path.join('logs-{}'.format(exp_name), 'taco_pretrained', 'checkpoint')
        sub_ckpt = ['tacotron_model.ckpt-{}'.format(x) for x in
                    range(int(args.start_step), int(args.end_step), int(args.syn_interval))]
        for ckpt_name in sub_ckpt:
            tmp_ckpt_meta_path = '{}.backup'.format(ckpt_meta_path)
            with open(ckpt_meta_path, 'rb') as fin, open(tmp_ckpt_meta_path, 'wb') as fout:
                meta_lines = fin.readlines()
                meta_lines[0] = 'model_checkpoint_path: "{}"\n'.format(ckpt_name).encode('utf-8')
                fout.writelines(meta_lines)
            os.remove(ckpt_meta_path)
            os.rename(tmp_ckpt_meta_path, ckpt_meta_path)

            if args.model == 'Tacotron':
                _ = tacotron_synthesize(args, hparams, taco_checkpoint, sentences)
            else:
                raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))

            tf.reset_default_graph()
            time.sleep(0.5)
            zip_tacotron_wavs(args)
            if args.rm:
                shutil.rmtree('tacotron_output')
    elif args.synthesis_mode == 'all_ckpt':
        exp_name = args.name
        ckpt_dir = os.path.join('logs-{}'.format(exp_name), 'taco_pretrained')
        ckpt_meta_path = os.path.join('logs-{}'.format(exp_name), 'taco_pretrained', 'checkpoint')
        all_ckpt = glob.glob(os.path.join(ckpt_dir, '*.meta'))
        all_ckpt = [os.path.basename(x)[:-5] for x in all_ckpt]
        for ckpt_name in all_ckpt:
            tmp_ckpt_meta_path = '{}.backup'.format(ckpt_meta_path)
            with open(ckpt_meta_path, 'rb') as fin, open(tmp_ckpt_meta_path, 'wb') as fout:
                meta_lines = fin.readlines()
                meta_lines[0] = 'model_checkpoint_path: "{}"\n'.format(ckpt_name).encode('utf-8')
                fout.writelines(meta_lines)
            os.remove(ckpt_meta_path)
            os.rename(tmp_ckpt_meta_path, ckpt_meta_path)

            if args.model == 'Tacotron':
                _ = tacotron_synthesize(args, hparams, taco_checkpoint, sentences)
            else:
                raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))

            tf.reset_default_graph()
            time.sleep(0.5)
            zip_tacotron_wavs(args)
            if args.rm:
                shutil.rmtree('tacotron_output')


if __name__ == '__main__':
    main()

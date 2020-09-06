# -*- coding: utf-8 -*-
# @Time    : 2019/4/11 17:33
# @Author  : MengnanChen
# @FileName: test_pd_file.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from datasets import audio
from hparams import hparams
from tacotron.utils.text import text_to_sequence


def split_func(x, split_pos):
    rst = []
    start = 0
    # x will be a numpy array with the contents of the placeholder below
    for i in range(split_pos.shape[0]):
        rst.append(x[:, start:start + split_pos[i]])
        start += split_pos[i]
    return rst


class TestPdFile:
    def load_pd(self, pd_file_path):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        graph_def = tf.GraphDef()
        with gfile.FastGFile(pd_file_path, 'rb') as fin:
            graph_def.ParseFromString(fin.read())
            tf.import_graph_def(graph_def, name='')
        self._hp = hparams

        self._pad = 0
        if self._hp.GL_on_GPU:
            # self.GLGPU_mel_inputs = tf.placeholder(tf.float32, (None, self._hp.num_mels), name='GLGPU_mel_inputs')
            self.GLGPU_lin_inputs = tf.placeholder(tf.float32, (None, self._hp.num_freq), name='GLGPU_lin_inputs')
            # self.GLGPU_mel_outputs = audio.inv_mel_spectrogram_tensorflow(self.GLGPU_mel_inputs, self._hp)
            self.GLGPU_lin_outputs = audio.inv_linear_spectrogram_tensorflow(self.GLGPU_lin_inputs, self._hp)

        output_node_name = 'Tacotron_model/inference/cbhg_linear_specs_projection/projection_cbhg_linear_specs_projection/BiasAdd:0'
        self.linear_outputs = self.sess.graph.get_tensor_by_name(output_node_name)

        self.inputs = self.sess.graph.get_tensor_by_name('inputs:0')
        self.input_lengths = self.sess.graph.get_tensor_by_name('input_lengths:0')
        self.split_infos = self.sess.graph.get_tensor_by_name('split_infos:0')
        self.p_inputs = tf.py_func(split_func, [self.inputs, self.split_infos[:, 0]], tf.int32)

    def inference(self, texts):
        cleaner_names = [x.strip() for x in self._hp.cleaners.split(',')]
        seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
        input_lengths = [len(seq) for seq in seqs]
        size_per_device = len(seqs)

        input_seqs = None
        split_infos = []
        for i in range(self._hp.tacotron_num_gpus):
            device_input = seqs[size_per_device * i: size_per_device * (i + 1)]
            device_input, max_seq_len = self._prepare_inputs(device_input)
            input_seqs = np.concatenate((input_seqs, device_input), axis=1) if input_seqs is not None else device_input
            split_infos.append([max_seq_len, 0, 0, 0])

        feed_dict = {
            self.inputs: input_seqs,
            self.input_lengths: np.asarray(input_lengths, dtype=np.int32),
            self.split_infos: np.asarray(split_infos, dtype=np.int32)
        }
        self.sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        linear = self.sess.run(self.linear_outputs, feed_dict=feed_dict)
        wav = self.sess.run(self.GLGPU_lin_outputs, feed_dict={self.GLGPU_lin_inputs: linear[0]})
        return wav

    def _prepare_inputs(self, inputs):
        max_len = max([len(x) for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)


def save_wavs(wav, filename, sample_rate):
    audio.save_wav(wav, filename, sample_rate)


def hanzi_to_pinyin(text):
    return text


if __name__ == '__main__':
    test = TestPdFile()
    test.load_pd('pd_file_path')
    texts = ['中国平安']
    texts = hanzi_to_pinyin(texts)
    wav = test.inference(texts)
    save_wavs(wav, 'test.wav', 22050)

# -*- coding: utf-8 -*-
# @Time    : 2019/4/29 15:34
# @Author  : MengnanChen
# @FileName: batch_griffin_lim.py
# @Software: PyCharm

'''
combine Tacotron2(Mel spectrum prediction network) and Tensorflow-Griffin_Lim(Vocoder) into one model
'''

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
from hparams import hparams


def _transform_ops():
    return [
        'add_default_attributes',
        'remove_nodes(op=Identity, op=CheckNumrics)',
        'fold_batch_norms',
        'fold_old_batch_norms',
        'strip_unused_nodes',
        'sort_by_execution_order'
    ]


def _get_node_name(tensors):
    if isinstance(tensors, list):
        return [tensor.name.split(':')[0] for tensor in tensors]
    else:
        return tensors.name.split(':')[0]


def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size


def _griffin_lim_tensorflow(S, hparams):
    '''TensorFlow implementation of Griffin-Lim
    Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
    '''
    with tf.variable_scope('griffinlim'):
        # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
        S = tf.expand_dims(S, 0)
        S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
        y = tf.contrib.signal.inverse_stft(S_complex, hparams.win_size, get_hop_size(hparams), hparams.n_fft)
        for i in range(hparams.griffin_lim_iters):
            est = tf.contrib.signal.stft(y, hparams.win_size, get_hop_size(hparams), hparams.n_fft)
            angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
            y = tf.contrib.signal.inverse_stft(S_complex * angles, hparams.win_size, get_hop_size(hparams),
                                               hparams.n_fft)
    return tf.squeeze(y, 0)


def _denormalize_tensorflow(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((tf.clip_by_value(D, -hparams.max_abs_value,
                                       hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (
                             2 * hparams.max_abs_value))
                    + hparams.min_level_db)
        else:
            return ((tf.clip_by_value(D, 0,
                                      hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (
                2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)


def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


###########################################################################################
# tensorflow Griffin-Lim
# Thanks to @begeekmyfriend: https://github.com/begeekmyfriend/Tacotron-2/blob/mandarin-new/datasets/audio.py

def inv_linear_spectrogram_tensorflow(spectrogram, hparams):
    '''Builds computational graph to convert spectrogram to waveform using TensorFlow.
    Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
    inv_preemphasis on the output after running the graph.
    '''
    D = _denormalize_tensorflow(spectrogram, hparams)

    S = tf.pow(_db_to_amp_tensorflow(D + hparams.ref_level_db), (1 / hparams.magnitude_power))
    return _griffin_lim_tensorflow(tf.pow(S, hparams.power), hparams)


###########################################################################################

with tf.Session(graph=tf.Graph()) as sess:
    max_linear_length = tf.placeholder(tf.int32, name='max_linear_length')
    GLGPU_lin_inputs = tf.placeholder(tf.float32, (None, None, hparams.num_freq), name='GLGPU_lin_inputs')
    input_width = tf.shape(GLGPU_lin_inputs)[1]
    real_width = tf.where(tf.greater(max_linear_length, input_width), input_width, max_linear_length)
    GLGPU_lin_inputs = tf.slice(GLGPU_lin_inputs, [0, 0, 0], [-1, real_width, -1])
    GLGPU_lin_outputs = inv_linear_spectrogram_tensorflow(GLGPU_lin_inputs, hparams)

    sess.run(tf.global_variables_initializer())
    print('GL output:', GLGPU_lin_outputs)

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                               ['griffinlim/inverse_stft_60/overlap_and_add/Reshape_2'])

    with tf.gfile.FastGFile('gf_model.pb', mode='wb') as fout:
        fout.write(constant_graph.SerializeToString())

    transformed_graph_def = TransformGraph(sess.graph.as_graph_def(),
                                           inputs=['GLGPU_lin_inputs', 'max_linear_length'],
                                           outputs=['griffinlim/inverse_stft_60/overlap_and_add/Reshape_2:0'],
                                           transforms=_transform_ops())

    constant_graph_def = graph_util.convert_variables_to_constants(
        sess,
        transformed_graph_def,
        ['griffinlim/inverse_stft_60/overlap_and_add/Reshape_2']
    )

    try:
        optimize_for_inference_lib.ensure_graph_is_valid(constant_graph_def)
        tf.train.write_graph(constant_graph_def, logdir='root/mengnan', name='final1/optimized_gf_with_length.pb',
                             as_text=False)
    except ValueError as e:
        print('Graph is invalid: {}'.format(e))

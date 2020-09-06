# -*- coding: utf-8 -*-
# @Time    : 2019/4/11 16:36
# @Author  : MengnanChen
# @FileName: freezer.py
# @Software: PyCharm

import os

import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import graph_util

from hparams import hparams
from tacotron.synthesize import Synthesizer as TacoSynthesizer


# refer to: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
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
        return [t.name.split(':')[0] for t in tensors]
    else:
        return tensors.name.split(':')[0]


def TacotronFreezer(checkpoint_dir, hp, output_dir):
    tf.reset_default_graph()
    synth = TacoSynthesizer()
    checkpoint_path = tf.train.get_checkpoint_state(checkpoint_dir).model_checkpoint_path
    synth.load(checkpoint_path, hp)

    os.makedirs(output_dir, exist_ok=True)

    tf.train.write_graph(synth.session.graph, output_dir, 'taco_variable_graph_def_ex.pd', as_text=False)

    transformed_graph_def = TransformGraph(
        synth.session.graph.as_graph_def(),
        inputs=_get_node_name([synth.model.inputs, synth.model.input_lengths, synth.model.split_infos]),
        outputs=_get_node_name(synth.model.tower_linear_outputs),
        transforms=_transform_ops()
    )
    const_graph_def = graph_util.convert_variables_to_constants(
        synth.session,
        transformed_graph_def,
        _get_node_name(synth.model.tower_linear_outputs)
    )

    try:
        optimize_for_inference_lib.ensure_graph_is_valid(const_graph_def)
        tf.train.write_graph(const_graph_def, output_dir, 'tacotron_optimized_frozen.pd', as_text=False)
    except Exception as e:
        print('Graph is invalid: {}'.format(e))


if __name__ == '__main__':
    checkpoint_dir = '/tacotron/model/'
    output_dir = 'tacotron'
    TacotronFreezer(checkpoint_dir, hparams, output_dir)

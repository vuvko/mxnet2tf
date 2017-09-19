import argparse
import json
import mxnet as mx
import tensorflow as tf
from tensorflow.python.framework import graph_util

from converter import Converter


def main(model_prefix, output_prefix, input_size=128):
    # Parsing JSON is easier because it contains operator name
    js_model = json.load(open(model_prefix + '-symbol.json', 'r'))
    mx_model, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
    params = arg_params
    params.update(aux_params)
    with tf.Session() as sess:
        tf_nodes = dict()
        # Workaround for input node
        input_data = tf.placeholder('float32', (None, input_size, input_size, 3), name='data')
        tf_nodes['data'] = input_data
        nodes = js_model['nodes']
        conv = Converter(tf_nodes, nodes, params)
        for node_idx, node in enumerate(nodes):
            op = node['op']
            print('Parsing node %s with operator %s and index %d' % (node['name'], op, node_idx))
            # Hack for older versions of MxNet
            if 'param' in node:
                node['attr'] = node['param']
            if op == 'BatchNorm':
                conv.create_bn(node)
            elif op == 'elemwise_add' or op == '_Plus':
                conv.create_elementwise(node)
            elif op == 'Activation':
                conv.create_activation(node)
            elif op == 'SoftmaxOutput':
                conv.create_softmax(node)
            elif op == 'Convolution':
                conv.create_conv(node)
            elif op == 'Pooling':
                conv.create_pooling(node)
            elif op == 'Flatten':
                conv.create_flatten(node)
            elif op == 'FullyConnected':
                conv.create_fc(node)
            elif op == 'L2Normalization':
                conv.create_norm(node)
        output_node_name = nodes[js_model['heads'][0][0]]['name']
        graph_def = sess.graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, [output_node_name])
        print('==================================')
        print('Input node:', tf_nodes['data'])
        print('Output node:', tf_nodes[output_node_name])
        print('Saving converted model to %s' % (output_prefix + '.pb'))
        with open(output_prefix + '.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('Saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mx_prefix', help='Prefix of pretrained mxnet model (with path)')
    parser.add_argument('tf_prefix', help='Name of TensorFlow model to save to (will be saved as binary protobuf)')
    parser.add_argument('-s', '--input_size', help='Size of network input', type=int, default=128)
    args = parser.parse_args()
    main(args.mx_prefix, args.tf_prefix, args.input_size)

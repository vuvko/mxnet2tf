import tensorflow as tf
import numpy as np


class Converter(object):

    def __init__(self, tf_nodes, mx_nodes, mx_params):
        self.tf_nodes = tf_nodes
        self.mx_nodes = mx_nodes
        self.mx_params = mx_params

    def to_tuple(self, string, conv_type=str):
        return tuple(map(conv_type, map(str.strip, string[1:-1].split(','))))

    def create_var(self, node, shape=None):
        node_name = node['name']
        if shape is None:
            if node_name in self.mx_params:
                shape = self.mx_params[node_name].shape
            else:
                shape = ()
        # print('Creating var with shape:', shape)
        created_node = tf.get_variable(node_name, shape=shape, initializer=tf.zeros_initializer)
        self.tf_nodes[node_name] = created_node
        # if node_name in params:
        #     tf_nodes[node_name].load(params[node_name].asnumpy())
        return created_node

    def create_bn(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        epsilon = float(node['attr']['eps'])
        input_shape = input_sym.get_shape()
        axis = list(range(len(input_shape) - 1))

        def create_bn_params(i):
            cur_node = self.mx_nodes[node['inputs'][i][0]]
            cur_name = cur_node['name']
            self.create_var(cur_node)
            self.tf_nodes[cur_name].load(self.mx_params[cur_name].asnumpy())
            return self.tf_nodes[cur_name]
        if len(node['inputs']) > 3:
            gamma, beta, mean, var = (create_bn_params(i) for i in range(1, 5))
        else:
            gamma, beta = (create_bn_params(i) for i in range(1, 3))
            mean = tf.get_variable(node_name + '_mean', shape=input_shape[-1], initializer=tf.zeros_initializer)
            mean.load(np.zeros((input_shape[-1],), dtype='float32'))
            var = tf.get_variable(node_name + '_var', shape=input_shape[-1], initializer=tf.ones_initializer)
            var.load(np.ones((input_shape[-1],), dtype='float32'))
        # TODO: add support for swtiching between train and inference phases
        # For inference use_global_stats=False is ignored
        #
        # if 'use_global_stats' in node['attr']:
        #     if node['attr']['use_global_stats'] == 'False':
        #         # print('Not use')
        #         mean, var = tf.nn.moments(input_sym, axis)
        # else:
        #     mean, var = tf.nn.moments(input_sym, axis)
        if 'fix_gamma' in node['attr']:
            if node['attr']['fix_gamma'] == 'True':
                # print('Fix')
                gamma = tf.get_variable(node_name + '_gamma_fixed', shape=input_shape[-1], initializer=tf.ones_initializer)
                gamma.load(np.ones((input_shape[-1],), dtype='float32'))
        else:
            gamma = tf.get_variable(node_name + '_gamma_fixed', shape=input_shape[-1], initializer=tf.ones_initializer)
            gamma.load(np.ones((input_shape[-1],), dtype='float32'))
        self.tf_nodes[node_name] = tf.nn.batch_normalization(input_sym, mean, var, beta, gamma, epsilon, name=node_name)
        return self.tf_nodes[node_name]

    def create_conv(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        num_filters_in = input_sym.get_shape()[-1]
        num_filters_out = int(node['attr']['num_filter'])
        kernel_size = self.to_tuple(node['attr']['kernel'], int)
        # TODO: add bias support
        # add_bias = node['attr']['no_bias'] != 'True'
        if 'num_group' in node['attr']:
            num_group = int(node['attr']['num_group'])
        else:
            num_group = 1
        if 'pad' in node['attr']:
            padding = self.to_tuple(node['attr']['pad'], int)
        else:
            padding = (0, 0)
        stride = self.to_tuple(node['attr']['stride'], int)
        weights_node = self.mx_nodes[node['inputs'][1][0]]
        weights = self.create_var(weights_node,
                             shape=(kernel_size[0], kernel_size[1], num_filters_in // num_group, num_filters_out))
        weights_numpy = self.mx_params[weights_node['name']].asnumpy().transpose((2, 3, 1, 0))
        if padding[0] > 0 or padding[1] > 0:
            padded_input = tf.pad(input_sym, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]], 'CONSTANT')
        else:
            padded_input = input_sym
        convolve = lambda input_sym, kernel, name=None: tf.nn.conv2d(input_sym, kernel, [1, stride[0], stride[1], 1], padding='VALID', name=name)
        weights.load(weights_numpy)
        if num_group > 1:
            input_groups = tf.split(axis=3, num_or_size_splits=num_group, value=padded_input)
            weight_groups = tf.split(axis=3, num_or_size_splits=num_group, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
            self.tf_nodes[node_name] = tf.concat(axis=3, values=output_groups, name=node_name)
        else:
            self.tf_nodes[node_name] = convolve(padded_input, weights, name=node_name)
        return self.tf_nodes[node_name]

    def create_pooling(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        pooling_type = node['attr']['pool_type']
        kernel_size = self.to_tuple(node['attr']['kernel'], int)
        if 'stride' in node['attr']:
            stride = self.to_tuple(node['attr']['stride'], int)
        else:
            stride = (1, 1)
        if 'global_pool' in node['attr']:
            global_pool = node['attr']['global_pool'] == 'True'
        else:
            global_pool = False
        if 'pad' in node['attr']:
            padding = self.to_tuple(node['attr']['pad'], int)
        else:
            padding = (0, 0)
        if global_pool:
            self.tf_nodes[node_name] = tf.reduce_mean(input_sym, reduction_indices=[1, 2], name=node_name)
        else:
            if padding[0] > 0 or padding[1] > 0:
                padded_input = tf.pad(input_sym,
                                      [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]],
                                      'CONSTANT')
            else:
                padded_input = input_sym
            if pooling_type == 'max':
                self.tf_nodes[node_name] = tf.nn.max_pool(padded_input,
                                                          ksize=[1, kernel_size[0], kernel_size[1], 1],
                                                          strides=[1, stride[0], stride[1], 1],
                                                          padding='VALID', name=node_name)
            else:
                raise NameError('Unknown pooling type: %s' % pooling_type)
        return self.tf_nodes[node_name]

    def create_activation(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        activation_type = node['attr']['act_type']
        # TODO: more activation types
        if activation_type == 'relu':
            activation_fn = tf.nn.relu
        else:
            raise NameError('Unknown activation type: %s' % activation_type)
        self.tf_nodes[node_name] = activation_fn(input_sym, name=node_name)
        return self.tf_nodes[node_name]

    def create_softmax(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.nn.softmax(input_sym, name=node_name)
        return self.tf_nodes[node_name]

    def create_elementwise(self, node, op='sum'):
        node_name = node['name']
        inputs_sym = [self.tf_nodes[self.mx_nodes[n[0]]['name']] for n in node['inputs']]
        # TODO: more elementwise types
        if op == 'sum':
            self.tf_nodes[node_name] = tf.add_n(inputs_sym, name=node_name)
        else:
            raise NameError('Unknown elementwise type: %s' % op)
        return self.tf_nodes[node_name]

    def create_fc(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        num_units_in = input_sym.get_shape()[1]
        num_units_out = int(node['attr']['num_hidden'])
        weights_node = self.mx_nodes[node['inputs'][1][0]]
        weights = self.create_var(weights_node, shape=(num_units_in, num_units_out))
        bias_node = self.mx_nodes[node['inputs'][2][0]]
        bias = self.create_var(bias_node, shape=(num_units_out,))
        weights_numpy = self.mx_params[weights_node['name']].asnumpy()
        weights.load(weights_numpy.T)
        bias.load(self.mx_params[bias_node['name']].asnumpy())
        self.tf_nodes[node_name] = tf.nn.xw_plus_b(input_sym, weights, bias, name=node_name)
        return self.tf_nodes[node_name]

    def create_norm(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.nn.l2_normalize(input_sym, dim=1, name=node_name)
        return self.tf_nodes[node_name]

    def create_flatten(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.contrib.layers.flatten(input_sym)
        return self.tf_nodes[node_name]

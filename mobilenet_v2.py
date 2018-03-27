"""
MobileNet v2.

As described in https://arxiv.org/abs/1801.04381

  Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple, OrderedDict

import functools
import tensorflow as tf

slim = tf.contrib.slim

Conv = namedtuple('Conv', ['kernel', 'stride', 'channel'])
InvertedBottleneck = namedtuple('InvertedBottleneck', ['up_sample', 'channel', 'stride', 'repeat'])

# Sequence of layers, described in Table 2
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, channel=32),              # first block, input 224x224x3
    InvertedBottleneck(up_sample=1, channel=16, stride=1, repeat=1),  # second block, input : 112x112x32
    InvertedBottleneck(up_sample=6, channel=24, stride=2, repeat=2),  # third block, input: 112x112x16
    InvertedBottleneck(up_sample=6, channel=32, stride=2, repeat=3),  # fourth block, input: 56x56x24
    InvertedBottleneck(up_sample=6, channel=64, stride=2, repeat=4),  # fifth block, input: 28x28x32
    InvertedBottleneck(up_sample=6, channel=96, stride=1, repeat=3),  # sixth block, input: 28x28x64
    InvertedBottleneck(up_sample=6, channel=160, stride=2, repeat=3),  # seventh block, input: 14x14x96
    InvertedBottleneck(up_sample=6, channel=320, stride=1, repeat=1),  # eighth block, input: 7x7x160
    Conv(kernel=[1, 1], stride=1, channel=1280),
    # AvgPool(kernel=[7, 7]),
    # Conv(kernel=[1, 1], stride=1, channel='num_class')
]


def mobilenet_v2_base(inputs,
                      final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      scope=None):
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = OrderedDict()

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    net = inputs
    with tf.variable_scope(scope, 'MobilenetV2', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
            for i, conv_def in enumerate(conv_defs):

                end_point = ''
                if isinstance(conv_def, Conv):
                    end_point = 'Conv2d_%d' % i
                    num_channel = depth(conv_def.channel)
                    net = slim.conv2d(net, num_channel, conv_def.kernel,
                                      activation_fn=tf.nn.relu6,
                                      stride=conv_def.stride,
                                      scope=end_point)
                    end_points[end_point] = net
                elif isinstance(conv_def, InvertedBottleneck):
                    stride = conv_def.stride

                    if conv_def.repeat <= 0:
                        raise ValueError('repeat value of inverted bottleneck should be greater than zero.')

                    for j in range(conv_def.repeat):
                        end_point = 'InvertedBottleneck_%d_%d' % (i, j)
                        prev_output = net
                        net = slim.conv2d(net, conv_def.up_sample * net.get_shape().as_list()[-1], [1, 1],
                                          activation_fn=tf.nn.relu6,
                                          scope=end_point + '_inverted_bottleneck')
                        end_points[end_point + '_inverted_bottleneck'] = net
                        net = slim.separable_conv2d(net, None, [3, 3],
                                                    depth_multiplier=1,
                                                    stride=stride,
                                                    activation_fn=tf.nn.relu6,
                                                    scope=end_point + '_dwise')
                        end_points[end_point + '_dwise'] = net

                        num_channel = depth(conv_def.channel)
                        net = slim.conv2d(net, num_channel, [1, 1],
                                          activation_fn=None,
                                          scope=end_point + '_linear')
                        end_points[end_point + '_linear'] = net

                        if stride == 1:
                            if prev_output.get_shape().as_list()[-1] != net.get_shape().as_list()[-1]:
                                # Assumption based on previous ResNet papers: If the number of filters doesn't match,
                                # there should be a conv 1x1 operation.
                                # reference(pytorch) : https://github.com/MG2033/MobileNet-V2/blob/master/layers.py#L29
                                prev_output = slim.conv2d(prev_output, num_channel, [1, 1],
                                                          activation_fn=None,
                                                          biases_initializer=None,
                                                          scope=end_point + '_residual_match')

                            # as described in Figure 4.
                            net = tf.add(prev_output, net, name=end_point + '_residual_add')
                            end_points[end_point + '_residual_add'] = net

                        stride = 1
                else:
                    raise ValueError('CONV_DEF is not valid.')

                if end_point == final_endpoint:
                    break

    return net, end_points


def mobilenet_v2_cls(inputs,
                     num_classes=1000,
                     dropout_keep_prob=0.999,
                     is_training=True,
                     min_depth=8,
                     depth_multiplier=1.0,
                     conv_defs=None,
                     prediction_fn=tf.contrib.layers.softmax,
                     reuse=None,
                     scope='MobilenetV2'):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))

    with tf.variable_scope(scope, 'MobilenetV2', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = mobilenet_v2_base(inputs, scope=scope,
                                                min_depth=min_depth,
                                                depth_multiplier=depth_multiplier,
                                                conv_defs=conv_defs)
            with tf.variable_scope('Logits'):
                # class
                if num_classes:
                    net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='Dropout_1')
                    # global pool
                    # Issue #1 : https://github.com/ildoonet/tf-mobilenet-v2/issues/1
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='Global_pool')
                    end_points['Global_pool'] = net

                    # classification
                    net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='Dropout_2')
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='Conv2d_1c_1x1')
                    net = slim.flatten(net, scope='Flatten')
                    end_points['Logits'] = net

                    if prediction_fn:
                        end_points['Predictions'] = prediction_fn(net, scope='Predictions')

        return net, end_points


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


mobilenet_v2_cls_075 = wrapped_partial(mobilenet_v2_cls, depth_multiplier=0.75)
mobilenet_v2_cls_050 = wrapped_partial(mobilenet_v2_cls, depth_multiplier=0.50)
mobilenet_v2_cls_025 = wrapped_partial(mobilenet_v2_cls, depth_multiplier=0.25)


def mobilenet_v2_arg_scope(is_training=True,
                           weight_decay=0.00004,
                           stddev=0.09,
                           regularize_depthwise=False):
    """Defines the default MobilenetV2 arg scope.
    Args:
      is_training: Whether or not we're training the model.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.
    Returns:
      An `arg_scope` to use for the mobilenet v2 model.
    """
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.9,
        'epsilon': 0.001,
        'fused': True,
        'zero_debias_moving_mean': True
    }

    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d], weights_regularizer=depthwise_regularizer) as sc:
                    return sc

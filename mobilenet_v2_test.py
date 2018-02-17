"""Tests for Mobilenet v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import mobilenet_v2

slim = tf.contrib.slim


class MobilenetV2Test(tf.test.TestCase):
    def testBuildClassificationNetwork(self):
        # test built network's output
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = mobilenet_v2.mobilenet_v2_cls(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith('MobilenetV2/Logits/Flatten'))
        self.assertListEqual(logits.get_shape().as_list(), [batch_size, num_classes])
        self.assertTrue('Predictions' in end_points)
        self.assertListEqual(end_points['Predictions'].get_shape().as_list(), [batch_size, num_classes])

    def testBuildPreLogitsNetwork(self):
        # test built network's output
        batch_size = 5
        height, width = 224, 224
        num_classes = None

        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = mobilenet_v2.mobilenet_v2_cls(inputs, num_classes)
        # TODO

    def testBuildAndCheck(self):
        # as described in table 1 & 2
        batch_size = 5
        num_classes = 1000
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            _, endpoints = mobilenet_v2.mobilenet_v2_cls(inputs, num_classes)
        endpoints_shapes = {
            'Conv2d_0': [batch_size, 112, 112, 32],
            'InvertedBottleneck_1_0_inverted_bottleneck': [batch_size, 112, 112, 32],
            'InvertedBottleneck_1_0_dwise': [batch_size, 112, 112, 32],
            'InvertedBottleneck_1_0_linear': [batch_size, 112, 112, 16],
            'InvertedBottleneck_1_0_residual_add': [batch_size, 112, 112, 16],

            'InvertedBottleneck_2_0_inverted_bottleneck': [batch_size, 112, 112, 96],
            'InvertedBottleneck_2_0_dwise': [batch_size, 56, 56, 96],
            'InvertedBottleneck_2_0_linear': [batch_size, 56, 56, 24],
            'InvertedBottleneck_2_1_inverted_bottleneck': [batch_size, 56, 56, 144],
            'InvertedBottleneck_2_1_dwise': [batch_size, 56, 56, 144],
            'InvertedBottleneck_2_1_linear': [batch_size, 56, 56, 24],
            'InvertedBottleneck_2_1_residual_add': [batch_size, 56, 56, 24],

            'InvertedBottleneck_3_0_inverted_bottleneck': [batch_size, 56, 56, 144],
            'InvertedBottleneck_3_0_dwise': [batch_size, 28, 28, 144],
            'InvertedBottleneck_3_0_linear': [batch_size, 28, 28, 32],
            'InvertedBottleneck_3_1_inverted_bottleneck': [batch_size, 28, 28, 192],
            'InvertedBottleneck_3_1_dwise': [batch_size, 28, 28, 192],
            'InvertedBottleneck_3_1_linear': [batch_size, 28, 28, 32],
            'InvertedBottleneck_3_1_residual_add': [batch_size, 28, 28, 32],
            'InvertedBottleneck_3_2_inverted_bottleneck': [batch_size, 28, 28, 192],
            'InvertedBottleneck_3_2_dwise': [batch_size, 28, 28, 192],
            'InvertedBottleneck_3_2_linear': [batch_size, 28, 28, 32],
            'InvertedBottleneck_3_2_residual_add': [batch_size, 28, 28, 32],

            'InvertedBottleneck_7_0_residual_add': [batch_size, 7, 7, 320],
            'Conv2d_8': [batch_size, 7, 7, 1280],
            'Global_pool': [batch_size, 1, 1, 1280],
            'Logits': [batch_size, num_classes],
            'Predictions': [batch_size, num_classes],
        }
        # self.assertItemsEqual(endpoints_shapes.keys(), endpoints.keys())
        for endpoint_name, expected_shape in endpoints_shapes.items():
            self.assertTrue(endpoint_name in endpoints)
            self.assertListEqual(endpoints[endpoint_name].get_shape().as_list(), expected_shape)

    def testModelHasExpectedNumberOfParameters(self):
        # See Table 4
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            mobilenet_v2.mobilenet_v2_base(inputs)
            total_params, _ = slim.model_analyzer.analyze_vars(
                slim.get_model_variables())
            self.assertAlmostEqual(2301200, total_params)

    def testBuildEndPointsWithDepthMultiplierLessThanOne(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = mobilenet_v2.mobilenet_v2_cls(inputs, num_classes)

        endpoint_keys = [key for key in end_points.keys()
                         if key.startswith('Conv') or key.startswith('Inverted')]

        _, end_points_with_multiplier = mobilenet_v2.mobilenet_v2_cls(
            inputs, num_classes, scope='depth_multiplied_net',
            depth_multiplier=0.5)

        for key in endpoint_keys:
            original_depth = end_points[key].get_shape().as_list()[3]
            new_depth = end_points_with_multiplier[key].get_shape().as_list()[3]
            self.assertEqual(0.5 * original_depth, new_depth)

    def testBuildEndPointsWithDepthMultiplierGreaterThanOne(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = mobilenet_v2.mobilenet_v2_cls(inputs, num_classes)

        endpoint_keys = [key for key in end_points.keys()
                         if key.startswith('Conv') or key.startswith('Inverted')]

        _, end_points_with_multiplier = mobilenet_v2.mobilenet_v2_cls(
            inputs, num_classes, scope='depth_multiplied_net',
            depth_multiplier=2.0)

        for key in endpoint_keys:
            original_depth = end_points[key].get_shape().as_list()[3]
            new_depth = end_points_with_multiplier[key].get_shape().as_list()[3]
            self.assertEqual(2.0 * original_depth, new_depth)

    def testRaiseValueErrorWithInvalidDepthMultiplier(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        with self.assertRaises(ValueError):
            _ = mobilenet_v2.mobilenet_v2_cls(
                inputs, num_classes, depth_multiplier=-0.1)
        with self.assertRaises(ValueError):
            _ = mobilenet_v2.mobilenet_v2_cls(
                inputs, num_classes, depth_multiplier=0.0)

    def testHalfSizeImages(self):
        batch_size = 5
        height, width = 112, 112
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = mobilenet_v2.mobilenet_v2_cls(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith('MobilenetV2/Logits'))
        self.assertListEqual(logits.get_shape().as_list(),
                             [batch_size, num_classes])

    def testUnknownImageShape(self):
        tf.reset_default_graph()
        batch_size = 2
        height, width = 300, 400
        num_classes = 1000
        input_np = np.random.uniform(0, 1, (batch_size, height, width, 3))
        with self.test_session() as sess:
            inputs = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3))
            logits, end_points = mobilenet_v2.mobilenet_v2_cls(inputs, num_classes)
            self.assertTrue(logits.op.name.startswith('MobilenetV2/Logits'))
            self.assertListEqual(logits.get_shape().as_list(), [batch_size, num_classes])
            pre_pool = end_points['Conv2d_8']
            feed_dict = {inputs: input_np}
            tf.global_variables_initializer().run()
            pre_pool_out = sess.run(pre_pool, feed_dict=feed_dict)
            self.assertListEqual(list(pre_pool_out.shape), [batch_size, 10, 13, 1280])

    def testUnknowBatchSize(self):
        batch_size = 1
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.placeholder(tf.float32, (None, height, width, 3))
        logits, _ = mobilenet_v2.mobilenet_v2_cls(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith('MobilenetV2/Logits'))
        self.assertListEqual(logits.get_shape().as_list(), [None, num_classes])
        images = tf.random_uniform((batch_size, height, width, 3))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(logits, {inputs: images.eval()})
            self.assertEquals(output.shape, (batch_size, num_classes))

    def testEvaluation(self):
        batch_size = 2
        height, width = 224, 224
        num_classes = 1000

        eval_inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, _ = mobilenet_v2.mobilenet_v2_cls(eval_inputs, num_classes,
                                                  is_training=False)
        predictions = tf.argmax(logits, 1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(predictions)
            self.assertEquals(output.shape, (batch_size,))

    def testTrainEvalWithReuse(self):
        train_batch_size = 5
        eval_batch_size = 2
        height, width = 150, 150
        num_classes = 1000

        train_inputs = tf.random_uniform((train_batch_size, height, width, 3))
        mobilenet_v2.mobilenet_v2_cls(train_inputs, num_classes)
        eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
        logits, _ = mobilenet_v2.mobilenet_v2_cls(eval_inputs, num_classes,
                                                  reuse=True)
        predictions = tf.argmax(logits, 1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(predictions)
            self.assertEquals(output.shape, (eval_batch_size,))

if __name__ == '__main__':
    tf.test.main()

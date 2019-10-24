import os
import sys
import logging
import fire
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorpack.dataflow import imgaug
from tensorpack.graph_builder import override_to_local_variable

from checkmate.checkmate import BestCheckpointSaver, get_best_checkpoint
from mobilenet_v2 import mobilenet_v2_arg_scope, mobilenet_v2_cls
from data_helper import get_imagenet_dataflow, GoogleNetResize, DATA_PER_EPOCH, DataFlowToQueue, get_augmentations
from train_helper import allreduce_grads, split_grad_list, merge_grad_list, get_post_init_ops

logger = logging.getLogger('Runner')
logger.setLevel(logging.DEBUG)
logger.propagate = False
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)

logger_s = logging.getLogger('tensorpack')
logger_s.setLevel(logging.WARNING)


#author_dir = '/data/public/ro/dataset/images/imagenet/ILSVRC/2012/object_localization/ILSVRC/Data/CLS-LOC'
author_dir = '/data/ILSVRC_2012'

class MobilenetRunner:
    def __init__(self):
        # hyperparameters & meta data
        self.__op_decay_steps_epoch = 1
        self.__interval_train_log_epoch = 0.1
        self.__interval_valid_log_epoch = 1.0

        # persistent tensorflow sessions
        self.persistent_sess = tf.Session(config=tf.ConfigProto())
        self.global_step = tf.Variable(0, trainable=False)
        self.global_step_add = tf.assign(self.global_step, self.global_step + 1)
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # tensors for training
        self.ph_train_image = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='image_train')
        self.ph_train_label = tf.placeholder(tf.int32, shape=(None, ), name='label_train')
        self.output_train = None
        self.loss_train = None
        self.acc_train_top1 = None
        self.acc_train_top5 = None
        self.optimizers = []
        self.optimize_op = None
        self.sync_op = None
        self.enqueue_threads = None
        self.learning_rate = None

        # tensors for validation
        self.ph_valid_image = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='image_valid')
        self.ph_valid_label = tf.placeholder(tf.int32, shape=(None,), name='label_valid')
        self.output_valid = None
        self.loss_valid = None
        self.acc_valid_top1 = None
        self.acc_valid_top5 = None
        self.ds_valid = None

    def __create_network_for_imagenet(self, ph_input, is_training, is_reuse, depth_multiplier=1.0):
        tensor_input = tf.divide(ph_input, 255.0, name='inp_divide')
        tensor_input = tf.subtract(tensor_input, 0.5, name='inp_subtract')
        tensor_input = tf.multiply(tensor_input, 2.0, name='inp_multiply')

        with slim.arg_scope(mobilenet_v2_arg_scope(is_training)):
            net, end_points = mobilenet_v2_cls(
                tensor_input,
                is_training=is_training,
                reuse=is_reuse,
                depth_multiplier=depth_multiplier
            )
            return net, end_points

    def __get_dataflow(self, is_train, batch, datadir):
        augmentors = get_augmentations(is_train)
        return get_imagenet_dataflow(datadir, 'train' if is_train else 'val', batch, augmentors)

    def train(self, datadir=author_dir, batch=64, max_epoch=250, num_gpu=1,
              depth_multiplier=1.0, learning_rate_init=0.0001, optimizer='rmsprop',
              model_path='/data/private/tf-mobilenet-v2-model/', checkpoint=None):
        assert os.path.exists(datadir), 'not exist datadir(%s)' % datadir
        assert batch > 0, 'batch should be larger than 0, batch=%d' % batch
        assert max_epoch > 0, 'max_epoch should be larger than 0, max_epoch=%d' % max_epoch
        assert num_gpu > 0, 'num_gpu should be larger than 0, max_epoch=%d' % num_gpu
        assert depth_multiplier > 0, 'depth_multiplier should be larger than 0, depth_multiplier=%d' % depth_multiplier
        assert learning_rate_init > 0, 'learning_rate_init should be larger than 0, learning_rate_init=%d' % learning_rate_init
        assert batch % num_gpu == 0, 'batch %d should be a multiplier of num_gpu=%d, %d' % (batch, num_gpu, batch % num_gpu)

        op_decay_steps = int(round(DATA_PER_EPOCH * self.__op_decay_steps_epoch / batch))
        op_decay_rate = 0.99    # TODO
        __interval_train_log = int(round(DATA_PER_EPOCH * self.__interval_train_log_epoch / batch))
        __interval_valid_log = int(round(DATA_PER_EPOCH * self.__interval_valid_log_epoch / batch))

        if self.output_train is None:
            # create dataflow & queue
            logger.info('creating a mobilenet graph...')
            train_dss = [self.__get_dataflow(is_train=True, batch=batch // num_gpu, datadir=datadir) for _ in range(num_gpu)]
            phs = [self.ph_train_image, self.ph_train_label]
            self.enqueue_threads = [DataFlowToQueue(train_dss[idx], phs, batch // num_gpu, queue_size=100) for idx in range(num_gpu)]

            # create optimizer
            self.learning_rate = tf.train.exponential_decay(
                learning_rate_init, self.global_step,
                decay_steps=op_decay_steps,
                decay_rate=op_decay_rate,
                staircase=True
            )

            logger.info('optimizer=%s' % optimizer)
            if optimizer == 'rmsprop':
                self.optimizers = [tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, momentum=0.9) for _ in range(num_gpu)]
            elif optimizer == 'momentum':
                self.optimizers = [tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9) for _ in range(num_gpu)]
            elif optimizer == 'sgd':
                self.optimizers = [tf.train.GradientDescentOptimizer(self.learning_rate) for _ in range(num_gpu)]
            elif optimizer == 'adam':
                self.optimizers = [tf.train.AdamOptimizer(self.learning_rate) for _ in range(num_gpu)]
            else:
                raise Exception('invalid optimizer name=%s' % optimizer)

            # create network graph for training
            logits = []
            losses = []
            grad_list = []
            train_image_batch = []
            train_label_batch = []
            for gpu_idx in range(num_gpu):
                logger.info('creating gpu tower @ %d' % (gpu_idx + 1))
                image_tensor, label_tensor = self.enqueue_threads[gpu_idx].dequeue()
                train_image_batch.append(image_tensor)
                train_label_batch.append(label_tensor)

                scope_name = 'tower%d' % gpu_idx
                with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_idx)), tf.variable_scope(scope_name):
                    logit, _ = self.__create_network_for_imagenet(
                        image_tensor,
                        is_training=self.is_training,
                        is_reuse=False,
                        depth_multiplier=depth_multiplier
                    )
                    logits.append(logit)
                    print(logit.shape)
                    print(label_tensor)

                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=label_tensor,
                        logits=logit
                    )
                    losses.append(loss)
                    loss_w_reg = tf.reduce_sum(loss) + tf.add_n(slim.losses.get_regularization_losses(scope=scope_name))

                    grad_list.append([x for x in self.optimizers[gpu_idx].compute_gradients(loss_w_reg) if x[0] is not None])

            self.output_train = tf.concat(logits, axis=0)
            train_image_batch = tf.concat(train_image_batch, axis=0)
            train_label_batch = tf.concat(train_label_batch, axis=0)

            # loss
            self.acc_train_top1 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(self.output_train, train_label_batch, k=1), dtype=tf.float32)
            )
            self.acc_train_top5 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(self.output_train, train_label_batch, k=5), dtype=tf.float32)
            )
            self.loss_train = tf.reduce_mean(losses)

            # use NCCL
            grads, all_vars = split_grad_list(grad_list)
            reduced_grad = allreduce_grads(grads, average=True)
            grads = merge_grad_list(reduced_grad, all_vars)

            # optimizer using NCCL
            train_ops = []
            for idx, grad_and_vars in enumerate(grads):
                # apply_gradients may create variables. Make them LOCAL_VARIABLESZ¸¸¸¸¸¸
                with tf.name_scope('apply_gradients'), tf.device(tf.DeviceSpec(device_type="GPU", device_index=idx)):
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='tower%d' % idx)
                    with tf.control_dependencies(update_ops):
                        train_ops.append(self.optimizers[idx].apply_gradients(grad_and_vars, name='apply_grad_{}'.format(idx)))

            self.optimize_op = tf.group(*train_ops, name='train_op')

        if self.output_valid is None:
            self.__create_validate(depth_multiplier, is_reuse=True)

        self.sync_op = get_post_init_ops()

        # weight saver : only tower0
        # saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tower0') + [self.global_step])
        saver = tf.train.Saver()
        best_ckpt_saver = BestCheckpointSaver(
            save_dir=model_path,
            num_to_keep=100,
            maximize=False,
            saver=saver
        )
        best_val_loss = 99999
        best_val_acc1 = 0
        best_val_acc5 = 0

        # training
        with self.persistent_sess.as_default():
            coord = tf.train.Coordinator()
            for idx in range(num_gpu):
                self.enqueue_threads[idx].set_coordinator(coord)
                self.enqueue_threads[idx].start()
            q_sizes = [self.enqueue_threads[idx].size() for idx in range(num_gpu)]
            self.persistent_sess.run(tf.global_variables_initializer())
            if checkpoint is None:
                logger.info('initialization - global variable init')
            elif checkpoint is 'latest':
                logger.info('initialization - restore from latest one')
                saver.restore(self.persistent_sess, tf.train.latest_checkpoint(model_path))
            else:
                logger.info('initialization - restore from model path')
                saver.restore(self.persistent_sess, model_path)
            self.persistent_sess.run(self.sync_op)
            logger.info('start to train... initial step=%d' % (self.persistent_sess.run(self.global_step) + 1))

            try:
                while True:
                    _, val_step = self.persistent_sess.run(
                        [self.optimize_op, self.global_step_add],
                        feed_dict={
                            self.is_training: True
                        }
                    )
                    if (val_step + 1) % 100 == 0:
                        val_loss, _, _, val_lr, val_acctop1, val_acctop5, val_q_size = self.persistent_sess.run([
                            self.loss_train, self.optimize_op, self.global_step_add, self.learning_rate,
                            self.acc_train_top1, self.acc_train_top5, q_sizes[0]
                        ], feed_dict={
                            self.is_training: True
                        })
                        logger.info('training epoch=%.1f/%d step=%d lr=%.6f loss=%.5f acc_top1=%.2f acc_top5=%.2f q=%d'
                                    % (
                                        (val_step + 1) * batch / DATA_PER_EPOCH,
                                        max_epoch,
                                        val_step + 1,
                                        val_lr,
                                        val_loss,
                                        val_acctop1, val_acctop5,
                                        val_q_size
                                    ))

                    if (val_step + 1) % __interval_valid_log == 0:
                        val_loss, acc_dict = self.validate(datadir, checkpoint=None, depth_multiplier=depth_multiplier)
                        logger.info('-- validation loss=%.5f acc_top1=%.5f acc_top5=%.5f' % (
                            val_loss,
                            acc_dict['top1'],
                            acc_dict['top5']
                        ))

                        # save & keep best model \wrt validation loss
                        best_ckpt_saver.handle(val_loss, self.persistent_sess, self.global_step)

                        if best_val_loss > val_loss:
                            best_val_loss = val_loss
                            best_val_acc1 = acc_dict['top1']
                            best_val_acc5 = acc_dict['top5']

                        # periodic synchronization
                        self.persistent_sess.run(self.sync_op)
                    if val_step > DATA_PER_EPOCH // batch * max_epoch:
                        break

                    assert val_step > 0
            except KeyboardInterrupt:
                logger.info('interrupted. stop training, saving...')
                saver.save(self.persistent_sess, os.path.join(model_path, 'model'), global_step=val_step)

        chk_path = get_best_checkpoint(model_path, select_maximum_value=False)
        logger.info('training done. best_model val_loss=%.5f top1=%.5f top5=%.5f ckpt=%s' % (
            best_val_loss, best_val_acc1, best_val_acc5, chk_path
        ))

    def __create_validate(self, depth_multiplier, is_reuse=False):
        # create network graph for validation
        logger.info('creating a mobilenet graph for validation... is_reuse=%d' % (is_reuse))
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)), tf.variable_scope('tower0'):
            self.output_valid, _ = self.__create_network_for_imagenet(
                self.ph_valid_image,
                is_training=self.is_training,
                is_reuse=is_reuse,
                depth_multiplier=depth_multiplier
            )

            # loss
            self.loss_valid = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.ph_valid_label,
                logits=self.output_valid
            )
        self.acc_valid_top1 = tf.cast(tf.nn.in_top_k(self.output_valid, self.ph_valid_label, k=1), dtype=tf.float32)
        self.acc_valid_top5 = tf.cast(tf.nn.in_top_k(self.output_valid, self.ph_valid_label, k=5), dtype=tf.float32)

    def validate(self, datadir=author_dir, batch=64, checkpoint=None, depth_multiplier=1.0):
        """
        :param datadir:
        :param batch:
        :param checkpoint: If checkpoint is provided, mobilenet graph will be loaded with the checkpoint. If not,
           the graph will reuse the exist weights(eg. training graph)
        :return: validation loss, accuracy dict
        """
        assert os.path.exists(datadir), 'not exist datadir(%s)' % datadir
        assert batch > 0, 'batch should be larger than 0, batch=%d' % batch
        assert (self.output_train is not None) if checkpoint is None else True, 'checkpoint is not provided when there are any exist graph.'

        if self.output_valid is None:
            self.__create_validate(depth_multiplier, (checkpoint is None))

        if self.ds_valid is None:
            self.ds_valid = self.__get_dataflow(is_train=False, batch=batch, datadir=datadir)
        with self.persistent_sess.as_default():
            is_reuse = checkpoint is None
            logger.debug('validate is_reuse=%d' % is_reuse)
            if is_reuse:
                # copy from tower0
                pass
            else:
                # TODO : load
                pass

            val_losses = []
            val_acctop1s = []
            val_acctop5s = []
            for img_batch, lb_batch in self.ds_valid.get_data():
                val_loss, val_acctop1, val_acctop5 = self.persistent_sess.run([
                    self.loss_valid, self.acc_valid_top1, self.acc_valid_top5
                ], feed_dict={
                    self.ph_valid_image: img_batch,
                    self.ph_valid_label: lb_batch,
                    self.is_training: False
                })

                # self.persistent_sess.run(self.loss_valid, feed_dict={self.ph_valid_image: img_batch,self.ph_valid_label: lb_batch,self.is_training: False})
                # self.persistent_sess.run(self.loss_train,feed_dict={self.ph_train_image: img_batch, self.ph_train_label: lb_batch,self.is_training: False})

                val_losses.extend(val_loss)
                val_acctop1s.extend(val_acctop1)
                val_acctop5s.extend(val_acctop5)
            return np.mean(val_losses), {
                'top1': np.mean(val_acctop1s),
                'top5': np.mean(val_acctop5s)
            }

    def inference(self, imagepath, checkpoint=None):
        pass


if __name__ == '__main__':
    fire.Fire(MobilenetRunner)

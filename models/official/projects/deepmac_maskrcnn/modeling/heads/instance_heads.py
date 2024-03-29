# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Instance prediction heads."""

# Import libraries

from typing import List, Union, Optional
from absl import logging
import tensorflow as tf

from official.modeling import tf_utils
from official.projects.deepmac_maskrcnn.modeling.heads import hourglass_network


class DeepMaskHead(tf.keras.layers.Layer):
    """Creates a mask head."""

    def __init__(self,
                 num_classes,
                 upsample_factor=2,
                 num_convs=4,
                 num_filters=256,
                 use_separable_conv=False,
                 activation='relu',
                 use_sync_bn=False,
                 norm_momentum=0.99,
                 norm_epsilon=0.001,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 class_agnostic=False,
                 convnet_variant='default',
                 crop_size: int = 14,
                 **kwargs):
        """Initializes a mask head.

        Args:
          num_classes: An `int` of the number of classes.
          upsample_factor: An `int` that indicates the upsample factor to generate
            the final predicted masks. It should be >= 1.
          num_convs: An `int` number that represents the number of the intermediate
            convolution layers before the mask prediction layers.
          num_filters: An `int` number that represents the number of filters of the
            intermediate convolution layers.
          use_separable_conv: A `bool` that indicates whether the separable
            convolution layers is used.
          activation: A `str` that indicates which activation is used, e.g. 'relu',
            'swish', etc.
          use_sync_bn: A `bool` that indicates whether to use synchronized batch
            normalization across different replicas.
          norm_momentum: A `float` of normalization momentum for the moving average.
          norm_epsilon: A `float` added to variance to avoid dividing by zero.
          kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
            Conv2D. Default is None.
          bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
          class_agnostic: A `bool`. If set, we use a single channel mask head that
            is shared between all classes.
          convnet_variant: A `str` denoting the architecture of network used in the
            head. Supported options are 'default', 'hourglass20', 'hourglass52'
            and 'hourglass100'.
          **kwargs: Additional keyword arguments to be passed.
        """
        super(DeepMaskHead, self).__init__(**kwargs)
        self._config_dict = {
            'num_classes': num_classes,
            'upsample_factor': upsample_factor,
            'num_convs': num_convs,
            'num_filters': num_filters,
            'use_separable_conv': use_separable_conv,
            'activation': activation,
            'use_sync_bn': use_sync_bn,
            'norm_momentum': norm_momentum,
            'norm_epsilon': norm_epsilon,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'class_agnostic': class_agnostic,
            'convnet_variant': convnet_variant,
            'crop_size': crop_size,
        }

        if tf.keras.backend.image_data_format() == 'channels_last':
            self._bn_axis = -1
        else:
            self._bn_axis = 1
        self._activation = tf_utils.get_activation(activation)

    def _get_conv_op_and_kwargs(self):
        conv_op = (tf.keras.layers.SeparableConv2D
                   if self._config_dict['use_separable_conv']
                   else tf.keras.layers.Conv2D)
        conv_kwargs = {
            'filters': self._config_dict['num_filters'],
            'kernel_size': 3,
            'padding': 'same',
        }
        if self._config_dict['use_separable_conv']:
            conv_kwargs.update({
                'depthwise_initializer': tf.keras.initializers.VarianceScaling(
                    scale=2, mode='fan_out', distribution='untruncated_normal'),
                'pointwise_initializer': tf.keras.initializers.VarianceScaling(
                    scale=2, mode='fan_out', distribution='untruncated_normal'),
                'bias_initializer': tf.zeros_initializer(),
                'depthwise_regularizer': self._config_dict['kernel_regularizer'],
                'pointwise_regularizer': self._config_dict['kernel_regularizer'],
                'bias_regularizer': self._config_dict['bias_regularizer'],
            })
        else:
            conv_kwargs.update({
                'kernel_initializer': tf.keras.initializers.VarianceScaling(
                    scale=2, mode='fan_out', distribution='untruncated_normal'),
                'bias_initializer': tf.zeros_initializer(),
                'kernel_regularizer': self._config_dict['kernel_regularizer'],
                'bias_regularizer': self._config_dict['bias_regularizer'],
            })

        return conv_op, conv_kwargs

    def _get_bn_op_and_kwargs(self):

        bn_op = (tf.keras.layers.experimental.SyncBatchNormalization
                 if self._config_dict['use_sync_bn']
                 else tf.keras.layers.BatchNormalization)
        bn_kwargs = {
            'axis': self._bn_axis,
            'momentum': self._config_dict['norm_momentum'],
            'epsilon': self._config_dict['norm_epsilon'],
        }

        return bn_op, bn_kwargs

    def build(self, input_shape):
        """Creates the variables of the head."""

        conv_op, conv_kwargs = self._get_conv_op_and_kwargs()
        # 根据 convnet_variant 标识，构建相应的 mask head
        self._build_convnet_variant(input_shape)

        self._deconv = tf.keras.layers.Conv2DTranspose(
            filters=self._config_dict['num_filters'],
            kernel_size=self._config_dict['upsample_factor'],
            strides=self._config_dict['upsample_factor'],
            padding='valid',
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2, mode='fan_out', distribution='untruncated_normal'),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=self._config_dict['kernel_regularizer'],
            bias_regularizer=self._config_dict['bias_regularizer'],
            name='mask-upsampling')

        bn_op, bn_kwargs = self._get_bn_op_and_kwargs()
        self._deconv_bn = bn_op(name='mask-deconv-bn', **bn_kwargs)

        if self._config_dict['class_agnostic']:
            num_filters = 1
        else:
            num_filters = self._config_dict['num_classes']

        conv_kwargs = {
            'filters': num_filters,
            'kernel_size': 1,
            'padding': 'valid',
        }
        if self._config_dict['use_separable_conv']:
            conv_kwargs.update({
                'depthwise_initializer': tf.keras.initializers.VarianceScaling(
                    scale=2, mode='fan_out', distribution='untruncated_normal'),
                'pointwise_initializer': tf.keras.initializers.VarianceScaling(
                    scale=2, mode='fan_out', distribution='untruncated_normal'),
                'bias_initializer': tf.zeros_initializer(),
                'depthwise_regularizer': self._config_dict['kernel_regularizer'],
                'pointwise_regularizer': self._config_dict['kernel_regularizer'],
                'bias_regularizer': self._config_dict['bias_regularizer'],
            })
        else:
            conv_kwargs.update({
                'kernel_initializer': tf.keras.initializers.VarianceScaling(
                    scale=2, mode='fan_out', distribution='untruncated_normal'),
                'bias_initializer': tf.zeros_initializer(),
                'kernel_regularizer': self._config_dict['kernel_regularizer'],
                'bias_regularizer': self._config_dict['bias_regularizer'],
            })
        self._mask_regressor = conv_op(name='mask-logits', **conv_kwargs)

        super(DeepMaskHead, self).build(input_shape)

    def call(self, inputs, training=None, afp: bool = None):
        """Forward pass of mask branch for the Mask-RCNN model.

        Args:
          inputs: A `list` of two tensors where
            inputs[0]: A `tf.Tensor` of shape [batch_size, num_instances,
              roi_height, roi_width, roi_channels], representing the ROI features.
            inputs[1]: A `tf.Tensor` of shape [batch_size, num_instances],
              representing the classes of the ROIs.
          training: A `bool` indicating whether it is in `training` mode.

        Returns:
          mask_outputs: A `tf.Tensor` of shape
            [batch_size, num_instances, roi_height * upsample_factor,
             roi_width * upsample_factor], representing the mask predictions.
        """
        roi_features, roi_classes = inputs
        # print("-------- Deep Mask Head info --------")
        # print("mask_head.afp:", afp)
        if True:
            # print("len(roi_features):", len(roi_features))
            features_shape = tf.shape(roi_features[0])
            batch_size, num_rois, height, width, filters = (
                features_shape[0], features_shape[1], features_shape[2],
                features_shape[3], features_shape[4])
            # print("height & width of roi_features:", height, width)
            if batch_size is None:
                batch_size = tf.shape(roi_features[0])[0]
            x = []
            for i in range(len(roi_features)):
                x.append(tf.reshape(roi_features[i], [-1, height, width, filters]))
            # print("len(x):", len(x))
        else:
            # print("afp:False")
            features_shape = tf.shape(roi_features)
            batch_size, num_rois, height, width, filters = (
                features_shape[0], features_shape[1], features_shape[2],
                features_shape[3], features_shape[4])
            if batch_size is None:
                batch_size = tf.shape(roi_features)[0]
            x = tf.reshape(roi_features, [-1, height, width, filters])

        # print("height & width of roi_features:", height, width)
        # print("filters:", filters)

        x = self._call_convnet_variant(x, afp=True)

        # logits_ff = []
        # if afp and isinstance(x, List):
        #     print("FF:True")
        #     x_ff = x[1]
        #     x = x[0]
        #     # x_ff 的 reshape 是否正确
        #     _, _, _, filters = x_ff.get_shape().as_list()
        #     # print("x_ff.shape:", tf.shape(x_ff))
        #     # print("x.shape:", tf.shape(x))
        #     # filters = tf.shape(x_ff)[-1]
        #     # print("x_ff.filters:", filters)
        #     x_ff = tf.reshape(x_ff, [-1, num_rois, height * width * filters])
        #     x_ff = self._fcs(x_ff)
        #     x_ff = self._fc_norms(x_ff)
        #     x_ff = self._activation(x_ff)
        #     logits_ff = x_ff
        #     # 维度为 (2, 10, 28*28)
        #     # print("logits_ff.shape after fcs:", tf.shape(logits_ff))
        # else:
        #     # print("FF:False")
        #     i = 0

        x = self._deconv(x)
        x = self._deconv_bn(x)
        x = self._activation(x)

        logits = self._mask_regressor(x)

        mask_height = height * self._config_dict['upsample_factor']
        mask_width = width * self._config_dict['upsample_factor']

        if self._config_dict['class_agnostic']:
            logits = tf.reshape(logits, [-1, num_rois, mask_height, mask_width, 1])
        else:
            logits = tf.reshape(
                logits,
                [-1, num_rois, mask_height, mask_width,
                 self._config_dict['num_classes']])
        # print("logits.shape before fusion:", tf.shape(logits))

        # if isinstance(logits_ff, tf.Tensor) and afp:
        #     # 怎么将 logits_ff 先 reshape 成 1类，再复制为 num_class 类，最后再相加融合？
        #     # x[1] = x[1].view(-1, 1, cfg.MRCNN.RESOLUTION, cfg.MRCNN.RESOLUTION)
        #     # x[1] = x[1].repeat(1, cfg.MODEL.NUM_CLASSES, 1, 1)
        #     # _, _, _, _, num_classes = logits.get_shape().as_list()
        #     logits_ff = tf.expand_dims(logits_ff, -1)
        #     logits_ff = tf.expand_dims(logits_ff, -1)
        #     logits_ff = tf.reshape(logits_ff, [-1, num_rois, mask_height, mask_width, 1])
        #     # print("logits_ff.shape before fusion:", tf.shape(logits_ff))
        #     logits = tf.add(logits, logits_ff)
        #     # print("logits.shape after fusion:", tf.shape(logits))
        #     # print("fusion:YES")
        # else:
        #     # print("fusion:NO")
        #     i = 1

        batch_indices = tf.tile(
            tf.expand_dims(tf.range(batch_size), axis=1), [1, num_rois])
        mask_indices = tf.tile(
            tf.expand_dims(tf.range(num_rois), axis=0), [batch_size, 1])

        if self._config_dict['class_agnostic']:
            class_gather_indices = tf.zeros_like(roi_classes, dtype=tf.int32)
        else:
            class_gather_indices = tf.cast(roi_classes, dtype=tf.int32)

        gather_indices = tf.stack(
            [batch_indices, mask_indices, class_gather_indices],
            axis=2)
        mask_outputs = tf.gather_nd(
            tf.transpose(logits, [0, 1, 4, 2, 3]), gather_indices)
        # print("-------------------------------------")
        return mask_outputs

    def _build_convnet_variant(self, input_shape):
        conv_op, conv_kwargs = self._get_conv_op_and_kwargs()
        # print("-------- DeepMaskHead._build_convnet_variant() --------")
        # print("input_shape[0]:", input_shape[0])
        # print("len(input_shape[0]):", len(input_shape[0]))
        # print("crop_size is:", self._config_dict['crop_size'])
        if isinstance(input_shape[0], List):
            num_levels = len(input_shape[0])
            # filters = input_shape[0][0][-1]
        else:
            num_levels = 1
            # filters = input_shape[0][-1]
        # old_filters = conv_kwargs['filters']
        # set_filters = self._config_dict['num_filters']
        # print("num_levels:", num_levels)
        # print("old filters:", old_filters)
        # print("input filters:", filters)
        # print("self._config_dict['num_filters']:", set_filters)
        # if filters != set_filters:
        #     conv_kwargs.update({'filters': filters})
        filters = self._config_dict['num_filters']
        variant = self._config_dict['convnet_variant']
        if variant == 'default':
            bn_op, bn_kwargs = self._get_bn_op_and_kwargs()
            # print("now conv_kwargs['filters']:", conv_kwargs['filters'])
            # ------------ conv_head + nomrs -------------#
            num_convs_start = 0
            if True: # isinstance(input_shape[0], List):
                num_convs_start = 1
                # print("mask_head._conv_head!")
                self._conv_head = []
                self._conv_head_norms = []
                for i in range(len(input_shape[0])):
                    conv_name = 'mask-conv-head_{}_{}'.format(0, i)
                    conv_op, conv_kwargs = self._get_conv_op_and_kwargs()
                    if 'kernel_initializer' in conv_kwargs:
                        # print("tf_utils.clone_initializer!")
                        conv_kwargs['kernel_initializer'] = tf_utils.clone_initializer(
                            conv_kwargs['kernel_initializer'])
                    self._conv_head.append(conv_op(name=conv_name, **conv_kwargs))
                    bn_name = 'mask-conv-head-bn_{}_{}'.format(0, i)
                    self._conv_head_norms.append(bn_op(name=bn_name, **bn_kwargs))

            bn_op, bn_kwargs = self._get_bn_op_and_kwargs()
            # ------------ convs + nomrs -------------#
            self._convs = []
            self._conv_norms = []
            for i in range(num_convs_start, self._config_dict['num_convs']):
                conv_name = 'mask-conv_{}'.format(i)
                conv_op, conv_kwargs = self._get_conv_op_and_kwargs()
                if 'kernel_initializer' in conv_kwargs:
                    # print("tf_utils.clone_initializer!")
                    conv_kwargs['kernel_initializer'] = tf_utils.clone_initializer(
                        conv_kwargs['kernel_initializer'])
                self._convs.append(conv_op(name=conv_name, **conv_kwargs))
                bn_name = 'mask-conv-bn_{}'.format(i)
                self._conv_norms.append(bn_op(name=bn_name, **bn_kwargs))

        elif variant == 'fully-connected':
            print("mask_head: fully-connected!")
            bn_op, bn_kwargs = self._get_bn_op_and_kwargs()
            # print("now conv_kwargs['filters']:", conv_kwargs['filters'])
            # ------------ conv_head + nomrs -------------#
            self._conv_head = []
            self._conv_head_norms = []
            for i in range(len(input_shape[0])):
                conv_name = 'mask-conv-head_{}_{}'.format(0, i)
                conv_op, conv_kwargs = self._get_conv_op_and_kwargs()
                self._conv_head.append(conv_op(name=conv_name, **conv_kwargs))
                bn_name = 'mask-conv-head-bn_{}_{}'.format(0, i)
                self._conv_head_norms.append(bn_op(name=bn_name, **bn_kwargs))

            # ------------ convs + nomrs -------------#
            self._convs = []
            self._conv_norms = []
            for i in range(1, self._config_dict['num_convs']):
                conv_name = 'mask-conv_{}'.format(i)
                conv_op, conv_kwargs = self._get_conv_op_and_kwargs()
                self._convs.append(conv_op(name=conv_name, **conv_kwargs))
                bn_name = 'mask-conv-bn_{}'.format(i)
                self._conv_norms.append(bn_op(name=bn_name, **bn_kwargs))

            # ------------ conv fc + nomrs -------------#
            # self._conv_fc = []
            # self._conv_fc_norms = []
            # conv_name = 'mask-conv-fc_{}'.format(0)
            # conv_op, conv_kwargs = self._get_conv_op_and_kwargs()
            # self._conv_fc.append(conv_op(name=conv_name, **conv_kwargs))
            # bn_name = 'mask-conv-bn-fc_{}'.format(0)
            # self._conv_fc_norms.append(bn_op(name=bn_name, **bn_kwargs))
            #
            # conv_op, conv_kwargs = self._get_conv_op_and_kwargs()
            # conv_kwargs.update({'filters': filters / 2})
            # # print("conv_kwargs['filters'] update for conv fc:", conv_kwargs['filters'])
            #
            # conv_name = 'mask-conv-fc_{}'.format(1)
            # self._conv_fc.append(conv_op(name=conv_name, **conv_kwargs))
            # bn_name = 'mask-conv-bn-fc_{}'.format(1)
            # self._conv_fc_norms.append(bn_op(name=bn_name, **bn_kwargs))
            #
            # conv_kwargs.update({'filters': filters})
            # # print("conv_kwargs['filters'] after conv fc:", conv_kwargs['filters'])

            # ------------ fc + nomrs -------------#
            fc_name = 'mask-fc_{}'.format(0)
            # print("self._config_dict['crop_size']:", self._config_dict['crop_size'])
            self._fcs = tf.keras.layers.Dense(
                units=(self._config_dict['crop_size'] * self._config_dict['upsample_factor']) ** 2,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=1 / 3.0, mode='fan_out', distribution='uniform'),
                kernel_regularizer=self._config_dict['kernel_regularizer'],
                bias_regularizer=self._config_dict['bias_regularizer'],
                name=fc_name)
            bn_name = 'mask-fc-bn_{}'.format(0)
            self._fc_norms = bn_op(name=bn_name, **bn_kwargs)

        elif variant == 'hourglass20':
            logging.info('Using hourglass 20 network.')
            # filters = self._config_dict['num_filters']
            self._hourglass = hourglass_network.hourglass_20(
                filters, initial_downsample=False, num_levels=num_levels)

        elif variant == 'hourglass52':
            logging.info('Using hourglass 52 network.')
            self._hourglass = hourglass_network.hourglass_52(
                filters, initial_downsample=False, num_levels=num_levels)

        elif variant == 'hourglass100':
            logging.info('Using hourglass 100 network.')
            self._hourglass = hourglass_network.hourglass_100(
                filters, initial_downsample=False, num_levels=num_levels)

        else:
            raise ValueError('Unknown ConvNet variant - {}'.format(variant))
        # print("-------------------------------------------------------")

    def _call_AFP_convnet(self, x, afp):
        if True: #afp:
            # ------------ Conv_head for each level -------------#
            # print("In AFP, len(x) is:", len(x))
            # print("len(self._conv_head):", len(self._conv_head))
            for i in range(len(x)):
                x[i] = self._conv_head[i](x[i])
                x[i] = self._conv_head_norms[i](x[i])
                x[i] = self._activation(x[i])
            # ------------ Fusion by max -------------#
            for i in range(1, len(x)):
                # x[0] = tf.maximum(x[0], x[i])
                # x[0] = tf.keras.layers.Maximum()([x[0], x[i]])
                # x[0] = tf.keras.layers.Add()([x[0], x[i]])
                x[0] = x[0] + x[i]
            x = x[0]
        return x

    def _call_convnet_variant(self, x, afp: bool = None):
        variant = self._config_dict['convnet_variant']
        if variant == 'default':
            # print("default.mask_head call!")
            x = self._call_AFP_convnet(x, afp)
            for conv, bn in zip(self._convs, self._conv_norms):
                x = conv(x)
                x = bn(x)
                x = self._activation(x)
            return x
        elif variant == 'fully-connected':
            x = self._call_AFP_convnet(x, afp)
            y = []
            # con2 - conv4
            for conv, bn in zip(self._convs, self._conv_norms):
                x = conv(x)
                x = bn(x)
                x = self._activation(x)
                y.append(x)
            x_fcn = y[-1]
            x_ff = y[1]
            # print("x_fcn.shape:", tf.shape(x_fcn))
            # print("x_ff.shape before conv fc:", tf.shape(x_ff))
            # conv4_fc + conv5_fc
            for conv, bn in zip(self._conv_fc, self._conv_fc_norms):
                x_ff = conv(x_ff)
                x_ff = bn(x_ff)
                x_ff = self._activation(x_ff)
            # print("x_ff.shape after conv fc:", tf.shape(x_ff))
            return [x_fcn, x_ff]
        elif variant == 'hourglass20':
            return self._hourglass(x, afp)[-1]
        elif variant == 'hourglass52':
            return self._hourglass(x, afp)[-1]
        elif variant == 'hourglass100':
            return self._hourglass(x, afp)[-1]
        else:
            raise ValueError('Unknown ConvNet variant - {}'.format(variant))

    def get_config(self):
        return self._config_dict

    @classmethod
    def from_config(cls, config):
        return cls(**config)

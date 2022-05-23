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

"""Contains the definitions of Feature Pyramid Networks (FPN)."""
from typing import Any, Mapping, Optional

# Import libraries
from absl import logging
import tensorflow as tf

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.modeling.decoders import factory
from official.vision.ops import spatial_transform_ops


@tf.keras.utils.register_keras_serializable(package='Vision')
class FPN(tf.keras.Model):
  """Creates a Feature Pyramid Network (FPN).

  This implemets the paper:
  Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, and
  Serge Belongie.
  Feature Pyramid Networks for Object Detection.
  (https://arxiv.org/pdf/1612.03144)
  """

  def __init__(
      self,
      input_specs: Mapping[str, tf.TensorShape],
      min_level: int = 3,
      max_level: int = 7,
      bpa: bool = False,
      num_filters: int = 256,
      fusion_type: str = 'sum',
      use_separable_conv: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a Feature Pyramid Network (FPN).

    Args:
      input_specs: A `dict` of input specifications. A dictionary consists of
        {level: TensorShape} from a backbone.
      min_level: An `int` of minimum level in FPN output feature maps.
      max_level: An `int` of maximum level in FPN output feature maps.
      num_filters: An `int` number of filters in FPN layers.
      fusion_type: A `str` of `sum` or `concat`. Whether performing sum or
        concat for feature fusion.
      use_separable_conv: A `bool`.  If True use separable convolution for
        convolution in FPN layers.
      activation: A `str` name of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_initializer: A `str` name of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    """
    self._config_dict = {
        'input_specs': input_specs,
        'min_level': min_level,
        'max_level': max_level,
        'bpa': bpa,
        'num_filters': num_filters,
        'fusion_type': fusion_type,
        'use_separable_conv': use_separable_conv,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }
    if use_separable_conv:
      conv2d = tf.keras.layers.SeparableConv2D
    else:
      conv2d = tf.keras.layers.Conv2D
    if use_sync_bn:
      norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      norm = tf.keras.layers.BatchNormalization
    activation_fn = tf.keras.layers.Activation(
        tf_utils.get_activation(activation))

    # Build input feature pyramid.
    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    # Get input feature pyramid from backbone.
    logging.info('FPN input_specs: %s', input_specs)
    inputs = self._build_input_pyramid(input_specs, min_level)
    backbone_max_level = min(int(max(inputs.keys())), max_level)
    
    print("-------- FPN info --------")
    print("min_level:", min_level)
    print("max_level:", max_level)
    print("backbone_max_level:", backbone_max_level)
    # Build lateral connections.
    # 其实就是构建一个 1*1 conv
    # range(3, 5+1)，C3-C5 经过一个 1*1 conv 存于 feats_lateral={"3":C3}
    feats_lateral = {}
    for level in range(min_level, backbone_max_level + 1):
      feats_lateral[str(level)] = conv2d(
          filters=num_filters,
          kernel_size=1,
          padding='same',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)(
              inputs[str(level)])

    # Build top-down path.
    # 取出 P5，它是由 C5 直接生成的
    feats = {str(backbone_max_level): feats_lateral[str(backbone_max_level)]}
    # range(5-1, 3-1, -1) -> 构造 P4-P3
    for level in range(backbone_max_level - 1, min_level - 1, -1):
      # feat_a 为 P3，经过 2x up上采样
      feat_a = spatial_transform_ops.nearest_upsampling(
          feats[str(level + 1)], 2)
      # feat_b 为 C2，经过 1*1 conv，即横向连接
      feat_b = feats_lateral[str(level)]
      # 两者相加得到 P2
      if fusion_type == 'sum':
        feats[str(level)] = feat_a + feat_b
      elif fusion_type == 'concat':
        feats[str(level)] = tf.concat([feat_a, feat_b], axis=-1)
      else:
        raise ValueError('Fusion type {} not supported.'.format(fusion_type))

    # TODO(xianzhi): consider to remove bias in conv2d.
    # Build post-hoc 3x3 convolution kernel.
    # # 分别经过一个 3*3 conv 用于消除上采样带来的混叠效应
    # range(3,5+1) -> P3-P5 
    for level in range(min_level, backbone_max_level + 1):
      feats[str(level)] = conv2d(
          filters=num_filters,
          strides=1,
          kernel_size=3,
          padding='same',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)(
              feats[str(level)])
    
    # add for bpa buttom-up path
    # bpa = False
    if bpa:
        print("bpa:True")
        # 取出 N3，它是由 P3 直接生成的
        # feats = {str(min_level): feats[str(min_level)]}
        for level in range(min_level+1, backbone_max_level+1):
            # feat_a 为 N3，经过 2x up下采样
            feat_a = conv2d(
                filters=num_filters,
                strides=2,
                kernel_size=3,
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer)(
                feats[str(level - 1)])
            # feat_b 为 P4
            feat_b = feats[str(level)]
            # 两者相加
            if fusion_type == 'sum':
                feats[str(level)] = feat_a + feat_b
            elif fusion_type == 'concat':
                feats[str(level)] = tf.concat([feat_a, feat_b], axis=-1)
            else:
                raise ValueError('Fusion type {} not supported.'.format(fusion_type))
            # 经过一个 1*1 conv 后得到 N4
            feats[str(level)] = conv2d(
                filters=num_filters,
                strides=1,
                kernel_size=3,
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer)(
                feats[str(level)])
    else:
        print("bpa:False")

    # TODO(xianzhi): consider to remove bias in conv2d.
    # Build coarser FPN levels introduced for RetinaNet.
    # range(5+1, 7+1) -> 构造 P6-P7
    # P6 由 P5 经过一个 3*3 conv 生成
    # P7 由 P6 先经过 ReLU 再经过一个 3*3 conv 生成
    # 最终得到 P7-P3
    for level in range(backbone_max_level + 1, max_level + 1):
      feats_in = feats[str(level - 1)]
      if level > backbone_max_level + 1:
        feats_in = activation_fn(feats_in)
      feats[str(level)] = conv2d(
          filters=num_filters,
          strides=2,
          kernel_size=3,
          padding='same',
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)(
              feats_in)

    # Apply batch norm layers.
    # range(3, 7+1) -> P3-P7 分别经过一个 BN
    for level in range(min_level, max_level + 1):
      feats[str(level)] = norm(
          axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
              feats[str(level)])
    print("len(feats):", len(feats))
    print("----------------------------------")

    self._output_specs = {
        str(level): feats[str(level)].get_shape()
        for level in range(min_level, max_level + 1)
    }

    super(FPN, self).__init__(inputs=inputs, outputs=feats, **kwargs)

  def _build_input_pyramid(self, input_specs: Mapping[str, tf.TensorShape],
                           min_level: int):
    assert isinstance(input_specs, dict)
    if min(input_specs.keys()) > str(min_level):
      raise ValueError(
          'Backbone min level should be less or equal to FPN min level')

    inputs = {}
    for level, spec in input_specs.items():
      inputs[level] = tf.keras.Input(shape=spec[1:])
    return inputs

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self) -> Mapping[str, tf.TensorShape]:
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_decoder_builder('fpn')
def build_fpn_decoder(
    input_specs: Mapping[str, tf.TensorShape],
    model_config: hyperparams.Config,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
) -> tf.keras.Model:
  """Builds FPN decoder from a config.

  Args:
    input_specs: A `dict` of input specifications. A dictionary consists of
      {level: TensorShape} from a backbone.
    model_config: A OneOfConfig. Model config.
    l2_regularizer: A `tf.keras.regularizers.Regularizer` instance. Default to
      None.

  Returns:
    A `tf.keras.Model` instance of the FPN decoder.

  Raises:
    ValueError: If the model_config.decoder.type is not `fpn`.
  """
  decoder_type = model_config.decoder.type
  decoder_cfg = model_config.decoder.get()
  if decoder_type != 'fpn':
    raise ValueError(f'Inconsistent decoder type {decoder_type}. '
                     'Need to be `fpn`.')
  norm_activation_config = model_config.norm_activation
  return FPN(
      input_specs=input_specs,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      pant=model_config.bpa,
      num_filters=decoder_cfg.num_filters,
      fusion_type=decoder_cfg.fusion_type,
      use_separable_conv=decoder_cfg.use_separable_conv,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

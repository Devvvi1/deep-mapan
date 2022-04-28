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

"""Tests for instance_heads.py."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.projects.deepmac_maskrcnn.modeling.heads import instance_heads as deep_instance_heads


class MaskHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      # (1, 1, False, False),
      # (1, 2, False, False),
      # (2, 1, False, False),
      (1, 2, 2, False, False),
      (2, 2, 2, False, True),
  )
  def test_forward(self, upsample_factor, num_convs, use_sync_bn, panet):
    print("---------------------------", num, "Test forward of Deep Mask Head---------------------------")
    mask_head = deep_instance_heads.DeepMaskHead(
        num_classes=3,
        upsample_factor=upsample_factor,
        num_convs=num_convs,
        num_filters=16,
        use_separable_conv=False,
        activation='relu',
        use_sync_bn=use_sync_bn,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    # roi_features = np.random.rand(2, 10, 14, 14, 16)
    if not panet:
        roi_features = np.random.rand(2, 10, 14, 14, 16)
    else:
        roi_features = []
        for i in range(3, 7+1):
            roi_features.append(np.random.rand(2, 10, 14, 14, 16))
    roi_classes = np.zeros((2, 10))
    masks = mask_head([roi_features, roi_classes])
    self.assertAllEqual(
        masks.numpy().shape,
        [2, 10, 14 * upsample_factor, 14 * upsample_factor])
    print("------------------------------------------------------------------------------------")

  def test_serialize_deserialize(self):
    print("---------------------------Test serialization of Deep Mask Head---------------------------")
    mask_head = deep_instance_heads.DeepMaskHead(
        num_classes=3,
        upsample_factor=2,
        num_convs=1,
        num_filters=256,
        use_separable_conv=False,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    config = mask_head.get_config()
    new_mask_head = deep_instance_heads.DeepMaskHead.from_config(config)
    self.assertAllEqual(
        mask_head.get_config(), new_mask_head.get_config())
    print("------------------------------------------------------------------------------------------")
  
  @parameterized.parameters(
      (1, 3, 'default', False),
      (2, 9, 'default', True),
      (3, 9, 'Fully-Connected', True),
  )
  def test_forward_class_agnostic(self, num, num_classes, convnet_variant, panet):
    print("---------------------------", num, "Test forward of class agnostic---------------------------")
    mask_head = deep_instance_heads.DeepMaskHead(
        num_classes=num_classes,
        class_agnostic=True,
        convnet_variant=convnet_variant
    )
    # roi_features = np.random.rand(2, 10, 14, 14, 16)
    if not panet:
        roi_features = np.random.rand(2, 10, 14, 14, 16)
    else:
        roi_features = []
        for i in range(3, 7+1):
            roi_features.append(np.random.rand(2, 10, 14, 14, 16))
    roi_classes = np.zeros((2, 10))
    masks = mask_head([roi_features, roi_classes])
    print("masks.shape:", masks.numpy().shape)
    self.assertAllEqual(masks.numpy().shape, [2, 10, 28, 28])
    print("------------------------------------------------------------------------------------")

  @parameterized.parameters(
      (1, False),
      (2, True),
  )
  def test_instance_head_hourglass(self, num, panet):
    print("---------------------------", num, "Test forward of HG-20---------------------------")
    mask_head = deep_instance_heads.DeepMaskHead(
        num_classes=3,
        class_agnostic=True,
        convnet_variant='hourglass20',
        num_filters=32,
        upsample_factor=2,
        crop_size=16
    )
    # roi_features = np.random.rand(2, 10, 16, 16, 16)
    if not panet:
        roi_features = np.random.rand(2, 10, 16, 16, 16)
    else:
        roi_features = []
        for i in range(3, 7+1):
            roi_features.append(np.random.rand(2, 10, 16, 16, 16))
    roi_classes = np.zeros((2, 10))
    masks = mask_head([roi_features, roi_classes])
    print("masks.shape:", masks.numpy().shape)
    self.assertAllEqual(masks.numpy().shape, [2, 10, 32, 32])
    print("---------------------------------------------------------------------------")


if __name__ == '__main__':
  tf.test.main()
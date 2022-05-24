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

"""Tests for distribution util functions."""

import sys
import tensorflow as tf

from official.common import distribute_utils

TPU_TEST = 'test_tpu' in sys.argv[0]


class DistributeUtilsTest(tf.test.TestCase):
  """Tests for distribute util functions."""

  def test_invalid_args(self):
    print("\n---------------------------test_invalid_args()---------------------------")
    with self.assertRaisesRegex(ValueError, '`num_gpus` can not be negative.'):
      _ = distribute_utils.get_distribution_strategy(num_gpus=-1)

    with self.assertRaisesRegex(ValueError,
                                '.*If you meant to pass the string .*'):
      _ = distribute_utils.get_distribution_strategy(
          distribution_strategy=False, num_gpus=0)
    with self.assertRaisesRegex(ValueError, 'When 2 GPUs are specified.*'):
      _ = distribute_utils.get_distribution_strategy(
          distribution_strategy='off', num_gpus=2)
    with self.assertRaisesRegex(ValueError,
                                '`OneDeviceStrategy` can not be used.*'):
      _ = distribute_utils.get_distribution_strategy(
          distribution_strategy='one_device', num_gpus=2)
    print("------------------------------------------------------------------------------------\n")

  def test_one_device_strategy_cpu(self):
    print("\n---------------------------test_one_device_strategy_cpu()---------------------------")
    ds = distribute_utils.get_distribution_strategy('one_device', num_gpus=0)
    self.assertEquals(ds.num_replicas_in_sync, 1)
    self.assertEquals(len(ds.extended.worker_devices), 1)
    self.assertIn('CPU', ds.extended.worker_devices[0])
    print("------------------------------------------------------------------------------------\n")

  def test_one_device_strategy_gpu(self):
    print("\n---------------------------test_one_device_strategy_gpu()---------------------------")
    ds = distribute_utils.get_distribution_strategy('one_device', num_gpus=1)
    self.assertEquals(ds.num_replicas_in_sync, 1)
    self.assertEquals(len(ds.extended.worker_devices), 1)
    self.assertIn('GPU', ds.extended.worker_devices[0])
    print("------------------------------------------------------------------------------------\n")

  # def test_mirrored_strategy(self):
  #   print("\n---------------------------test_mirrored_strategy()---------------------------")
  #   # CPU only.
  #   _ = distribute_utils.get_distribution_strategy(num_gpus=0)
  #   # 5 GPUs.
  #   ds = distribute_utils.get_distribution_strategy(num_gpus=5)
  #   self.assertEquals(ds.num_replicas_in_sync, 5)
  #   self.assertEquals(len(ds.extended.worker_devices), 5)
  #   for device in ds.extended.worker_devices:
  #     self.assertIn('GPU', device)
  #
  #   _ = distribute_utils.get_distribution_strategy(
  #       distribution_strategy='mirrored',
  #       num_gpus=2,
  #       all_reduce_alg='nccl',
  #       num_packs=2)
  #   with self.assertRaisesRegex(
  #       ValueError,
  #       'When used with `mirrored`, valid values for all_reduce_alg are.*'):
  #     _ = distribute_utils.get_distribution_strategy(
  #         distribution_strategy='mirrored',
  #         num_gpus=2,
  #         all_reduce_alg='dummy',
  #         num_packs=2)
  #   print("------------------------------------------------------------------------------------\n")

  # def test_mwms(self):
  #   print("\n---------------------------test_mwms()---------------------------")
  #   distribute_utils.configure_cluster(worker_hosts=None, task_index=-1)
  #   ds = distribute_utils.get_distribution_strategy(
  #       'multi_worker_mirrored', all_reduce_alg='nccl')
  #   self.assertIsInstance(
  #       ds, tf.distribute.experimental.MultiWorkerMirroredStrategy)
  #
  #   with self.assertRaisesRegex(
  #       ValueError,
  #       'When used with `multi_worker_mirrored`, valid values.*'):
  #     _ = distribute_utils.get_distribution_strategy(
  #         'multi_worker_mirrored', all_reduce_alg='dummy')
  #   print("------------------------------------------------------------------------------------\n")

  # def test_no_strategy(self):
  #   print("\n---------------------------test_no_strategy()---------------------------")
  #   ds = distribute_utils.get_distribution_strategy('off')
  #   self.assertIs(ds, tf.distribute.get_strategy())
  #   print("------------------------------------------------------------------------------------\n")

  def test_tpu_strategy(self):
    print("\n---------------------------test_tpu_strategy()---------------------------")
    if not TPU_TEST:
      self.skipTest('Only Cloud TPU VM instances can have local TPUs.')
    with self.assertRaises(ValueError):
      _ = distribute_utils.get_distribution_strategy('tpu')

    ds = distribute_utils.get_distribution_strategy('tpu', tpu_address='local')
    self.assertIsInstance(
        ds, tf.distribute.TPUStrategy)
    print("------------------------------------------------------------------------------------\n")

  def test_invalid_strategy(self):
    print("\n---------------------------test_invalid_strategy()---------------------------")
    with self.assertRaisesRegexp(
        ValueError,
        'distribution_strategy must be a string but got: False. If'):
      distribute_utils.get_distribution_strategy(False)
    with self.assertRaisesRegexp(
        ValueError, 'distribution_strategy must be a string but got: 1'):
      distribute_utils.get_distribution_strategy(1)
    print("------------------------------------------------------------------------------------\n")

  def test_get_strategy_scope(self):
    print("\n---------------------------test_get_strategy_scope()---------------------------")
    ds = distribute_utils.get_distribution_strategy('one_device', num_gpus=0)
    with distribute_utils.get_strategy_scope(ds):
      self.assertIs(tf.distribute.get_strategy(), ds)
    with distribute_utils.get_strategy_scope(None):
      self.assertIsNot(tf.distribute.get_strategy(), ds)
    print("------------------------------------------------------------------------------------\n")

if __name__ == '__main__':
  tf.test.main()

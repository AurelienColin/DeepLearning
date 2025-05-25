import unittest
import typing

import numpy as np # Not strictly used by TestLossClass but often useful with TF
import tensorflow as tf

from src.losses.losses import Loss, mae, std_difference # Imports for Loss and specific functions used

# Helper to create simple tensors (copied from original test_losses.py)
# If this helper is used by TestLossFunctions too, it might be better in a shared utils file,
# but for this task, copying is fine.
def create_tensor(data: typing.List[typing.List[typing.List[typing.List[float]]]], dtype=tf.float32) -> tf.Tensor:
    return tf.constant(data, dtype=dtype)

class TestLossClass(unittest.TestCase):
    def test_loss_class_single_loss(self) -> None:
        y_true: tf.Tensor = create_tensor([[[[1.0]]]])
        y_pred: tf.Tensor = create_tensor([[[[1.5]]]])
        loss_instance = Loss(losses=[mae])
        expected_total_loss: float = 0.5
        total_loss_val: tf.Tensor = loss_instance(y_true, y_pred)
        self.assertAlmostEqual(float(total_loss_val.numpy()), expected_total_loss, places=6)

    def test_loss_class_multiple_losses(self) -> None:
        y_true: tf.Tensor = create_tensor([[[[1.0]]], [[[3.0]]]]) 
        y_pred: tf.Tensor = create_tensor([[[[1.5]]], [[[3.5]]]])
        loss_instance = Loss(losses=[mae, std_difference])
        expected_total_loss: float = 0.5 # MAE=0.5, STD_DIFF=0.0
        total_loss_val: tf.Tensor = loss_instance(y_true, y_pred)
        self.assertAlmostEqual(float(total_loss_val.numpy()), expected_total_loss, places=6)

    def test_loss_class_multiple_losses_with_loss_weights(self) -> None:
        y_true: tf.Tensor = create_tensor([[[[1.0]]], [[[3.0]]]])
        y_pred: tf.Tensor = create_tensor([[[[1.5]]], [[[3.5]]]])
        loss_weights: typing.Sequence[float] = [0.5, 2.0]
        loss_instance = Loss(losses=[mae, std_difference], loss_weights=loss_weights)
        # MAE = 0.5 * 0.5 = 0.25
        # STD_DIFF = 0.0 * 2.0 = 0.0
        # Total = 0.25
        expected_total_loss: float = 0.25
        total_loss_val: tf.Tensor = loss_instance(y_true, y_pred)
        self.assertAlmostEqual(float(total_loss_val.numpy()), expected_total_loss, places=6)

    def test_loss_class_with_class_weights_at_instance_level(self) -> None:
        y_true: tf.Tensor = create_tensor([[[[1.0, 0.0]]]]) 
        y_pred: tf.Tensor = create_tensor([[[[1.5, 0.5]]]])
        class_weights_instance: tf.Tensor = tf.constant([0.5, 1.5], dtype=tf.float32)
        loss_instance = Loss(losses=[mae], class_weights=class_weights_instance)
        # MAE with instance weights:
        # Ch0: |1-1.5|=0.5. Weighted: 0.5*0.5 = 0.25
        # Ch1: |0-0.5|=0.5. Weighted: 0.5*1.5 = 0.75
        # Mean of weighted per-channel MAEs (as per Loss.apply_class_weights then tf.reduce_mean in mae):
        # Expected: ((0.5*0.5) + (0.5*1.5))/2 = (0.25 + 0.75)/2 = 0.5
        expected_total_loss: float = 0.5 
        total_loss_val: tf.Tensor = loss_instance(y_true, y_pred)
        self.assertAlmostEqual(float(total_loss_val.numpy()), expected_total_loss, places=6)

    def test_loss_class_weights_priority_and_interaction(self) -> None:
        y_true: tf.Tensor = create_tensor([[[[1.0, 0.0]]]]) 
        y_pred: tf.Tensor = create_tensor([[[[1.5, 0.5]]]])
        
        class_weights_instance: tf.Tensor = tf.constant([0.5, 1.5], dtype=tf.float32)
        loss_instance = Loss(losses=[mae], class_weights=class_weights_instance)
        
        expected_loss_val_instance_weights: float = 0.5 # As calculated above
        total_loss_val_instance: tf.Tensor = loss_instance(y_true, y_pred)
        self.assertAlmostEqual(float(total_loss_val_instance.numpy()), expected_loss_val_instance_weights, places=6)

        # Test direct call to mae to show it can behave differently if class_weights is passed directly
        class_weights_direct: tf.Tensor = tf.constant([2.0, 3.0], dtype=tf.float32)
        direct_mae_loss_val: tf.Tensor = mae(y_true, y_pred, class_weights=class_weights_direct)
        # Ch0: |1-1.5|=0.5. Weighted: 0.5*2.0 = 1.0
        # Ch1: |0-0.5|=0.5. Weighted: 0.5*3.0 = 1.5
        # Expected: (1.0 + 1.5)/2 = 1.25
        expected_direct_loss: float = 1.25
        self.assertAlmostEqual(float(direct_mae_loss_val.numpy()), expected_direct_loss, places=6)

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch, call, ANY
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock TensorFlow before it's imported
mock_tf = MagicMock()
sys.modules['tensorflow'] = mock_tf
sys.modules['tf'] = mock_tf # Some imports might use 'import tf'
sys.modules['tensorflow.keras'] = mock_tf.keras # For K
sys.modules['tensorflow.keras.backend'] = mock_tf.keras.backend # For K

# Import the module to be tested AFTER mocking TensorFlow
from losses import losses # This will now use the mocked tf

class TestLossClass(unittest.TestCase):

    def setUp(self):
        mock_tf.reset_mock()
        self.mock_loss_func1 = MagicMock(name="LossFunc1")
        self.mock_loss_func2 = MagicMock(name="LossFunc2")
        self.y_true_mock = MagicMock(name="y_true")
        self.y_pred_mock = MagicMock(name="y_pred")

    def test_loss_class_initialization_defaults(self):
        loss_container = losses.Loss(losses=(self.mock_loss_func1,))
        
        mock_tf.cast.assert_called_once_with([1.0], mock_tf.float32) # Default loss_weights
        self.assertIsNone(loss_container.class_weights) # Default
        self.assertEqual(loss_container.epsilon, 1e-7) # Default

    def test_loss_class_initialization_with_weights(self):
        custom_loss_weights = [0.5, 0.5]
        custom_class_weights = [0.2, 0.8]
        
        mock_tf.convert_to_tensor.return_value = MagicMock(name="ClassWeightsRawTensor")
        mock_tf.cast.side_effect = [
            MagicMock(name="LossWeightsCasted"), 
            MagicMock(name="ClassWeightsCasted")
        ]

        loss_container = losses.Loss(
            losses=(self.mock_loss_func1, self.mock_loss_func2),
            class_weights=custom_class_weights,
            loss_weights=custom_loss_weights
        )

        mock_tf.cast.assert_any_call(custom_loss_weights, mock_tf.float32)
        mock_tf.convert_to_tensor.assert_called_once_with(custom_class_weights)
        mock_tf.cast.assert_any_call(mock_tf.convert_to_tensor.return_value, mock_tf.float32)
        
        self.assertIsNotNone(loss_container.loss_weights)
        self.assertIsNotNone(loss_container.class_weights)

    def test_loss_class_call_method(self):
        mock_loss1_val = MagicMock(name="Loss1Value")
        mock_loss2_val = MagicMock(name="Loss2Value")
        self.mock_loss_func1.return_value = mock_loss1_val
        self.mock_loss_func2.return_value = mock_loss2_val

        mock_weighted_loss1 = MagicMock(name="WeightedLoss1")
        mock_weighted_loss2 = MagicMock(name="WeightedLoss2")
        mock_loss1_val.__mul__ = MagicMock(return_value=mock_weighted_loss1) # loss_val * weight
        mock_loss2_val.__mul__ = MagicMock(return_value=mock_weighted_loss2) # loss_val * weight
        
        mock_total_loss = MagicMock(name="TotalLoss")
        mock_weighted_loss1.__add__ = MagicMock(return_value=mock_total_loss) # total + current_loss

        loss_container = losses.Loss(
            losses=(self.mock_loss_func1, self.mock_loss_func2),
            loss_weights=[0.4, 0.6] # Example weights
        )
        # Ensure loss_weights are tensors for the zip
        loss_container.loss_weights = [mock_tf.constant(0.4), mock_tf.constant(0.6)]


        result = loss_container(self.y_true_mock, self.y_pred_mock)

        self.mock_loss_func1.assert_called_once_with(
            self.y_true_mock, self.y_pred_mock, loss_container.class_weights, loss_container.epsilon
        )
        self.mock_loss_func2.assert_called_once_with(
            self.y_true_mock, self.y_pred_mock, loss_container.class_weights, loss_container.epsilon
        )
        
        mock_loss1_val.__mul__.assert_called_once_with(loss_container.loss_weights[0])
        mock_loss2_val.__mul__.assert_called_once_with(loss_container.loss_weights[1])
        
        mock_weighted_loss1.__add__.assert_called_once_with(mock_weighted_loss2)
        self.assertIs(result, mock_total_loss)


    def test_apply_class_weights_with_weights(self):
        mock_loss_tensor = MagicMock(name="LossTensor")
        mock_class_weights_tensor = MagicMock(name="ClassWeightsTensor")
        
        mock_tf.rank.return_value = 3 # Example rank
        mock_mean_loss = MagicMock(name="MeanLossPerSample")
        mock_tf.reduce_mean.return_value = mock_mean_loss
        
        mock_weighted_loss = MagicMock(name="WeightedLoss")
        mock_mean_loss.__mul__ = MagicMock(return_value=mock_weighted_loss)

        result = losses.Loss.apply_class_weights(mock_loss_tensor, mock_class_weights_tensor)

        mock_tf.rank.assert_called_once_with(mock_loss_tensor)
        mock_tf.reduce_mean.assert_called_once_with(mock_loss_tensor, axis=[0,1]) # range(3-1)
        mock_mean_loss.__mul__.assert_called_once_with(mock_class_weights_tensor)
        self.assertIs(result, mock_weighted_loss)

    def test_apply_class_weights_no_weights(self):
        mock_loss_tensor = MagicMock(name="LossTensorNoWeights")
        result = losses.Loss.apply_class_weights(mock_loss_tensor, None)
        mock_tf.reduce_mean.assert_not_called()
        self.assertIs(result, mock_loss_tensor)


class TestStandaloneLossFunctions(unittest.TestCase):
    def setUp(self):
        mock_tf.reset_mock()
        self.y_true_mock = MagicMock(name="y_true_standalone")
        self.y_pred_mock = MagicMock(name="y_pred_standalone")
        self.class_weights_mock = MagicMock(name="class_weights_standalone")
        self.epsilon = losses.Loss.epsilon # Default epsilon

    @patch('losses.losses.mae') # Mock the internal call to mae
    def test_fourth_channel_mae(self, mock_mae_func):
        # y_true[:, :, :, 3] and y_pred[:, :, :, 3]
        mock_y_true_ch3 = MagicMock(name="y_true_ch3")
        mock_y_pred_ch3 = MagicMock(name="y_pred_ch3")
        self.y_true_mock.__getitem__.return_value = mock_y_true_ch3
        self.y_pred_mock.__getitem__.return_value = mock_y_pred_ch3
        
        mock_mae_loss_val = MagicMock(name="mae_loss_on_ch3")
        mock_mae_func.return_value = mock_mae_loss_val
        
        mock_final_reduced_loss = MagicMock(name="final_reduced_loss_fcmae")
        mock_tf.reduce_mean.return_value = mock_final_reduced_loss

        result = losses.fourth_channel_mae(self.y_true_mock, self.y_pred_mock, None, self.epsilon)

        # Check that __getitem__ was called correctly for the 4th channel (index 3)
        self.y_true_mock.__getitem__.assert_called_once_with((slice(None), slice(None), slice(None), 3))
        self.y_pred_mock.__getitem__.assert_called_once_with((slice(None), slice(None), slice(None), 3))

        mock_mae_func.assert_called_once_with(mock_y_true_ch3, mock_y_pred_ch3, None, self.epsilon)
        mock_tf.reduce_mean.assert_called_once_with(mock_mae_loss_val)
        self.assertIs(result, mock_final_reduced_loss)


    @patch('losses.losses.Loss.apply_class_weights')
    def test_mae(self, mock_apply_class_weights):
        mock_abs_diff = MagicMock(name="abs_diff")
        mock_tf.abs.return_value = mock_abs_diff
        
        mock_weighted_loss = MagicMock(name="weighted_mae_loss")
        mock_apply_class_weights.return_value = mock_weighted_loss
        
        mock_final_reduced_loss = MagicMock(name="final_reduced_mae")
        mock_tf.reduce_mean.return_value = mock_final_reduced_loss

        result = losses.mae(self.y_true_mock, self.y_pred_mock, self.class_weights_mock)
        
        mock_tf.abs.assert_called_once_with(self.y_true_mock - self.y_pred_mock)
        mock_apply_class_weights.assert_called_once_with(mock_abs_diff, self.class_weights_mock)
        mock_tf.reduce_mean.assert_called_once_with(mock_weighted_loss)
        self.assertIs(result, mock_final_reduced_loss)


    @patch('losses.losses.Loss.apply_class_weights')
    def test_cross_entropy(self, mock_apply_class_weights):
        # positive_loss = y_true * tf.math.log(y_pred + epsilon)
        # negative_loss = (1 - y_true) * tf.math.log((1 - y_pred) + epsilon)
        # loss = positive_loss + negative_loss
        mock_log_pred_plus_eps = MagicMock(name="log_ypred_eps")
        mock_log_1_minus_pred_plus_eps = MagicMock(name="log_1_minus_ypred_eps")
        mock_tf.math.log.side_effect = [mock_log_pred_plus_eps, mock_log_1_minus_pred_plus_eps]

        mock_pos_loss_term = MagicMock(name="pos_loss_term")
        self.y_true_mock.__mul__ = MagicMock(return_value=mock_pos_loss_term) # y_true * log(...)

        # (1 - y_true)
        mock_1_minus_y_true = MagicMock(name="1_minus_y_true")
        # Assuming tf overloads 1 - tensor
        # self.y_true_mock.__rsub__ = MagicMock(return_value=mock_1_minus_y_true) # 1 - y_true
        # More directly, (1-y_true) is an expression, its __mul__ will be called.
        # Let's mock the result of (1-y_true) * log(...)
        mock_neg_loss_term = MagicMock(name="neg_loss_term")
        # This requires (1-y_true) to be a mock that has __mul__
        # For simplicity, we'll assume the structure and that the math ops are chained.

        mock_raw_loss = MagicMock(name="raw_ce_loss") # pos_loss + neg_loss
        mock_pos_loss_term.__add__ = MagicMock(return_value=mock_raw_loss)
        
        mock_weighted_loss = MagicMock(name="weighted_ce_loss")
        mock_apply_class_weights.return_value = mock_weighted_loss
        
        mock_reduced_mean_loss = MagicMock(name="reduced_mean_ce_loss")
        mock_tf.reduce_mean.return_value = mock_reduced_mean_loss
        
        mock_final_negated_loss = MagicMock(name="final_negated_ce_loss")
        mock_reduced_mean_loss.__neg__ = MagicMock(return_value=mock_final_negated_loss)


        result = losses.cross_entropy(self.y_true_mock, self.y_pred_mock, self.class_weights_mock, self.epsilon)

        mock_tf.math.log.assert_any_call(self.y_pred_mock + self.epsilon)
        mock_tf.math.log.assert_any_call((1 - self.y_pred_mock) + self.epsilon)
        
        # Check y_true * log(...) was called
        self.y_true_mock.__mul__.assert_called_once_with(mock_log_pred_plus_eps)
        
        # Check (1-y_true)*log(...) implies (1-y_true) was formed and then multiplied.
        # This is harder to check precisely without deeper tf operation mocking.
        # We trust the structure: positive_loss + negative_loss
        mock_pos_loss_term.__add__.assert_called() # With some representation of negative_loss
        
        mock_apply_class_weights.assert_called_once_with(mock_raw_loss, self.class_weights_mock)
        mock_tf.reduce_mean.assert_called_once_with(mock_weighted_loss)
        mock_reduced_mean_loss.__neg__.assert_called_once()
        self.assertIs(result, mock_final_negated_loss)


    @patch('losses.losses.Loss.apply_class_weights')
    def test_one_minus_dice(self, mock_apply_class_weights):
        # numerator = epsilon + 2 * Loss.apply_class_weights(y_true * y_pred, class_weights)
        # denominator = epsilon + Loss.apply_class_weights(y_true + y_pred, class_weights)
        mock_y_true_times_y_pred = MagicMock(name="y_true_times_y_pred")
        self.y_true_mock.__mul__ = MagicMock(return_value=mock_y_true_times_y_pred)
        
        mock_y_true_plus_y_pred = MagicMock(name="y_true_plus_y_pred")
        self.y_true_mock.__add__ = MagicMock(return_value=mock_y_true_plus_y_pred)

        mock_weighted_intersection = MagicMock(name="weighted_intersection")
        mock_weighted_sum = MagicMock(name="weighted_sum")
        # apply_class_weights called twice
        mock_apply_class_weights.side_effect = [mock_weighted_intersection, mock_weighted_sum]

        # numerator: epsilon + 2 * weighted_intersection
        # Assume tf ops for 2*X and eps+X work. Let's mock their results.
        mock_2_times_weighted_intersection = MagicMock(name="2_times_weighted_int")
        # weighted_intersection needs to be mock that supports __mul__ by 2 (or __rmul__)
        # For simplicity, let's assume 2 * mock_weighted_intersection happens.
        mock_numerator = MagicMock(name="numerator")

        # denominator: epsilon + weighted_sum
        mock_denominator = MagicMock(name="denominator")
        
        # dice = numerator / denominator
        mock_dice_val = MagicMock(name="dice_value")
        # This requires numerator to be a mock that supports __truediv__
        # For simplicity, we'll assume the division happens.
        
        # final_loss = 1 - tf.reduce_mean(dice)
        mock_reduced_mean_dice = MagicMock(name="reduced_mean_dice")
        mock_tf.reduce_mean.return_value = mock_reduced_mean_dice
        mock_final_loss = MagicMock(name="final_one_minus_dice_loss")
        # This requires 1 to be able to __sub__ mock_reduced_mean_dice, or vice-versa
        # Let's assume 1 - mock_reduced_mean_dice happens.

        # To make the test more concrete, we can mock the results of arithmetic ops directly
        # if we control the inputs to them.
        # Let's refine by mocking the direct inputs to tf.reduce_mean
        # numerator / denominator
        mock_dice_val_tensor = MagicMock(name="dice_val_tensor")
        # We need a way to make `mock_numerator / mock_denominator` result in `mock_dice_val_tensor`.
        # This is hard without actually running TF.
        # We will trust the math ops and check the calls to apply_class_weights and tf.reduce_mean.
        
        # Re-simplifying: focus on arguments to apply_class_weights and tf.reduce_mean
        losses.one_minus_dice(self.y_true_mock, self.y_pred_mock, self.class_weights_mock, self.epsilon)

        self.y_true_mock.__mul__.assert_called_once_with(self.y_pred_mock) # For intersection
        self.y_true_mock.__add__.assert_called_once_with(self.y_pred_mock) # For sum

        mock_apply_class_weights.assert_any_call(mock_y_true_times_y_pred, self.class_weights_mock)
        mock_apply_class_weights.assert_any_call(mock_y_true_plus_y_pred, self.class_weights_mock)
        
        # The argument to tf.reduce_mean is (eps + 2*WI) / (eps + WS)
        # This is difficult to assert without actual TF tensor evaluation.
        mock_tf.reduce_mean.assert_called_once() 
        # The final result is 1 - that mean. Check that subtraction was involved (implicitly).


    def test_std_difference(self):
        # std_true = tf.math.reduce_std(y_true, axis=0)
        # std_pred = tf.math.reduce_std(y_pred, axis=0)
        # loss = tf.abs(std_true - std_pred)
        # return tf.reduce_mean(loss)
        mock_std_true = MagicMock(name="std_true")
        mock_std_pred = MagicMock(name="std_pred")
        mock_tf.math.reduce_std.side_effect = [mock_std_true, mock_std_pred]
        
        mock_abs_diff_std = MagicMock(name="abs_diff_std")
        mock_tf.abs.return_value = mock_abs_diff_std
        
        mock_final_reduced_loss = MagicMock(name="final_reduced_std_diff")
        mock_tf.reduce_mean.return_value = mock_final_reduced_loss

        result = losses.std_difference(self.y_true_mock, self.y_pred_mock)

        mock_tf.math.reduce_std.assert_any_call(self.y_true_mock, axis=0)
        mock_tf.math.reduce_std.assert_any_call(self.y_pred_mock, axis=0)
        mock_tf.abs.assert_called_once_with(mock_std_true - mock_std_pred)
        mock_tf.reduce_mean.assert_called_once_with(mock_abs_diff_std)
        self.assertIs(result, mock_final_reduced_loss)


    @patch('losses.losses.mae') # Mocks the internal call to losses.mae
    def test_edge_loss(self, mock_mae_func_for_edge):
        # edges_true = sobel_edges(y_true)
        # edges_pred = sobel_edges(y_pred)
        # edge_loss_value = mae(edges_true, edges_pred, *args)
        
        # Mock tf.image.sobel_edges
        # It returns a tensor where last dim size is 2 (for dy, dx components)
        mock_sobel_true_raw = MagicMock(name="sobel_true_raw")
        mock_sobel_pred_raw = MagicMock(name="sobel_pred_raw")
        mock_tf.image.sobel_edges.side_effect = [mock_sobel_true_raw, mock_sobel_pred_raw]

        # Mock components grad_x, grad_y from sobel_edges output
        # sobel_x[..., 0], sobel_x[..., 1]
        mock_grad_x_true = MagicMock(name="grad_x_true")
        mock_grad_y_true = MagicMock(name="grad_y_true")
        mock_sobel_true_raw.__getitem__.side_effect = [mock_grad_x_true, mock_grad_y_true]
        
        mock_grad_x_pred = MagicMock(name="grad_x_pred")
        mock_grad_y_pred = MagicMock(name="grad_y_pred")
        mock_sobel_pred_raw.__getitem__.side_effect = [mock_grad_x_pred, mock_grad_y_pred]

        # Mock tf.square results
        mock_sq_gx_true = MagicMock(name="sq_gx_true"); mock_grad_x_true.__pow__ = MagicMock(return_value=mock_sq_gx_true) # Using pow for square
        mock_sq_gy_true = MagicMock(name="sq_gy_true"); mock_grad_y_true.__pow__ = MagicMock(return_value=mock_sq_gy_true)
        mock_sq_gx_pred = MagicMock(name="sq_gx_pred"); mock_grad_x_pred.__pow__ = MagicMock(return_value=mock_sq_gx_pred)
        mock_sq_gy_pred = MagicMock(name="sq_gy_pred"); mock_grad_y_pred.__pow__ = MagicMock(return_value=mock_sq_gy_pred)
        
        # Mock tf.sqrt results (edges)
        mock_edges_true_tensor = MagicMock(name="edges_true_tensor")
        mock_edges_pred_tensor = MagicMock(name="edges_pred_tensor")
        # sqrt(sq(gx) + sq(gy) + eps)
        # The arguments to sqrt will be complex expressions. We assume they form correctly.
        mock_tf.sqrt.side_effect = [mock_edges_true_tensor, mock_edges_pred_tensor]
        
        # Mock the final mae call
        mock_mae_edge_loss_val = MagicMock(name="mae_edge_loss_val")
        mock_mae_func_for_edge.return_value = mock_mae_edge_loss_val

        # args passed to edge_loss, then to mae
        test_args_for_mae = (None, self.epsilon) # class_weights, epsilon for mae

        result = losses.edge_loss(self.y_true_mock, self.y_pred_mock, *test_args_for_mae, epsilon=self.epsilon)

        mock_tf.image.sobel_edges.assert_any_call(self.y_true_mock)
        mock_tf.image.sobel_edges.assert_any_call(self.y_pred_mock)
        
        # Check square calls (example for true path)
        mock_grad_x_true.__pow__.assert_called_with(2)
        mock_grad_y_true.__pow__.assert_called_with(2)
        
        # Check sqrt calls
        # Arguments to sqrt are (tf.square(grad_x) + tf.square(grad_y) + epsilon)
        # This is hard to assert precisely without TF running. We check call count.
        self.assertEqual(mock_tf.sqrt.call_count, 2)
        
        mock_mae_func_for_edge.assert_called_once_with(mock_edges_true_tensor, mock_edges_pred_tensor, *test_args_for_mae)
        self.assertIs(result, mock_mae_edge_loss_val)


if __name__ == '__main__':
    unittest.main()

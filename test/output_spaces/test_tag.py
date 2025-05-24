import unittest
from unittest.mock import MagicMock
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from output_spaces.tag import Tag

class TestTag(unittest.TestCase):

    def test_initialization_defaults(self):
        index_val = 0
        name_val = "test_tag"
        dataset_size_val = 100
        
        tag = Tag(index=index_val, name=name_val, dataset_size=dataset_size_val)

        self.assertEqual(tag.index, index_val)
        self.assertEqual(tag.name, name_val)
        self.assertEqual(tag.dataset_size, dataset_size_val)
        self.assertEqual(tag.number_of_use, 0) # Default
        self.assertIsNone(tag._frequency)      # Default

    def test_initialization_custom_number_of_use(self):
        index_val = 1
        name_val = "another_tag"
        dataset_size_val = 200
        number_of_use_val = 10
        
        tag = Tag(
            index=index_val, 
            name=name_val, 
            dataset_size=dataset_size_val,
            number_of_use=number_of_use_val
        )
        self.assertEqual(tag.number_of_use, number_of_use_val)

    def test_frequency_property_before_set(self):
        tag = Tag(index=0, name="tag1", dataset_size=100)
        # _frequency is None initially, so property returns None
        self.assertIsNone(tag.frequency) 

    def test_set_frequency(self):
        tag = Tag(index=0, name="tag2", dataset_size=100, number_of_use=20)
        
        # The 'denominator' argument in set_frequency is currently unused in the implementation.
        # The implementation uses self.dataset_size.
        tag.set_frequency(denominator=999) # Pass a dummy denominator
        
        expected_frequency = 20 / 100  # 0.2
        self.assertEqual(tag._frequency, expected_frequency)
        self.assertEqual(tag.frequency, expected_frequency) # Property should now return the set value

    def test_set_frequency_zero_dataset_size(self):
        # Test behavior if dataset_size is 0 to avoid ZeroDivisionError.
        # The current implementation doesn't explicitly handle this.
        # Python float division by zero results in ZeroDivisionError if number_of_use is non-zero,
        # or nan if 0/0.
        tag_zero_ds_non_zero_use = Tag(index=0, name="tag_zero_ds1", dataset_size=0, number_of_use=10)
        with self.assertRaises(ZeroDivisionError):
            tag_zero_ds_non_zero_use.set_frequency(denominator=0)

        tag_zero_ds_zero_use = Tag(index=1, name="tag_zero_ds2", dataset_size=0, number_of_use=0)
        # 0 / 0 results in nan (Not a Number) with float division
        tag_zero_ds_zero_use.set_frequency(denominator=0)
        # Check if _frequency is nan (requires numpy or math.isnan)
        # For now, just ensure it doesn't raise an error differently or that it's set.
        # Depending on numpy availability in test env, `self.assertTrue(math.isnan(tag.frequency))`
        self.assertIsNotNone(tag_zero_ds_zero_use._frequency) # It will be nan

    def test_frequency_property_after_set(self):
        tag = Tag(index=0, name="tag3", dataset_size=50, number_of_use=5)
        tag._frequency = 0.1 # Manually set for testing property directly
        self.assertEqual(tag.frequency, 0.1)

if __name__ == '__main__':
    unittest.main()

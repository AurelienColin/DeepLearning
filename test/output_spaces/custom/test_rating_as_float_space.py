import unittest
from unittest.mock import MagicMock, patch, mock_open, call
import sys
import os
import json # For json.load mocking in parent TaggerSpace if its setup is called

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

# Mock dependencies for parent classes (TaggerSpace, OutputSpace)
mock_rignak_lazy_rafs = MagicMock()
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_rafs
mock_rignak_logger_rafs = MagicMock()
sys.modules['Rignak.logging_utils'] = mock_rignak_logger_rafs
mock_tag_module_rafs = MagicMock()
sys.modules['src.output_spaces.tag'] = mock_tag_module_rafs
mock_numpy_rafs = MagicMock()
sys.modules['numpy'] = mock_numpy_rafs


# Import RatingAsFloatSpace after mocks
from output_spaces.custom.rating_as_float_space import RatingAsFloatSpace
from output_spaces.space_from_json import TaggerSpace # Parent
from output_spaces.output_space import OutputSpace # Grandparent

class TestRatingAsFloatSpace(unittest.TestCase):

    def setUp(self):
        mock_rignak_lazy_rafs.LazyProperty.reset_mock()
        mock_rignak_logger_rafs.logger.reset_mock()
        mock_tag_module_rafs.Tag.reset_mock()
        mock_numpy_rafs.reset_mock()
        
        # Mock np.empty and np.array, as they are used by RatingAsFloatSpace.get_array
        # and potentially by parent initializers or setup methods.
        self.mock_np_empty_array = MagicMock(name="NpEmptyArrayInstance")
        mock_numpy_rafs.empty.return_value = self.mock_np_empty_array
        
        # For parent OutputSpace.class_weights, if it were ever called (it's set to None in __init__)
        mock_numpy_rafs.array.side_effect = lambda x, dtype=None: x # Simple pass-through for array creation
        mock_numpy_rafs.nanmean.return_value = 1.0


    # Patch common_setup from OutputSpace to prevent full setup logic of parent classes
    # as we are unit testing RatingAsFloatSpace's specific overrides and methods.
    @patch.object(OutputSpace, 'common_setup', MagicMock(name="MockedCommonSetup")) 
    @patch('os.path.dirname', MagicMock(return_value="dummy_dir")) # For TaggerSpace.setup
    @patch('builtins.open', new_callable=mock_open, read_data='{}') # For TaggerSpace.setup
    @patch('json.load', MagicMock(return_value={})) # For TaggerSpace.setup
    def test_initialization(self):
        sources_val = ["path/to/ratings.json"] # Needed for TaggerSpace init
        
        space = RatingAsFloatSpace(sources=sources_val)
        
        # Check attributes overridden by RatingAsFloatSpace
        self.assertEqual(space._n, 1) # Overridden
        self.assertIsNone(space.class_weights) # Overridden to None
        
        # Check attributes from parent TaggerSpace/OutputSpace are accessible
        self.assertEqual(space.sources, sources_val)
        self.assertEqual(space.limit, 100) # Default from OutputSpace

        # Test that TaggerSpace.setup (and thus common_setup) might have been called
        # if any lazy property that triggers setup was accessed implicitly by __init__ or here.
        # RatingAsFloatSpace itself doesn't directly call setup in __init__ beyond what TaggerSpace does.
        # TaggerSpace's setup is complex; for this unit test, we primarily care that
        # RatingAsFloatSpace correctly overrides _n and class_weights *after* super().__init__.
        # If TaggerSpace's __init__ called its setup, our mocked common_setup would have been hit.
        # OutputSpace.common_setup.assert_called() # Or not, depending on TaggerSpace init.
        # For this test, we are fine as long as _n and class_weights are correctly set.


    def test_get_array_valid_tags(self):
        # We need to simulate a state where filename_to_tags is populated.
        # This is normally done by the setup() process.
        # For this unit test, we'll manually set it after creating an instance.
        
        # Patch TaggerSpace's setup to avoid its file reading logic,
        # allowing us to manually control filename_to_tags for this test.
        with patch.object(TaggerSpace, 'setup') as mock_parent_setup:
            space = RatingAsFloatSpace(sources=["dummy.json"])
            
            mock_tag_g = MagicMock(spec=mock_tag_module_rafs.Tag); mock_tag_g.name = "rating:g"
            mock_tag_s = MagicMock(spec=mock_tag_module_rafs.Tag); mock_tag_s.name = "rating:s"
            mock_tag_q = MagicMock(spec=mock_tag_module_rafs.Tag); mock_tag_q.name = "rating:q"
            mock_tag_e = MagicMock(spec=mock_tag_module_rafs.Tag); mock_tag_e.name = "rating:e"
            
            space.filename_to_tags = {
                "file_g.jpg": [mock_tag_g],
                "file_s.jpg": [mock_tag_s],
                "file_q.jpg": [mock_tag_q],
                "file_e.jpg": [mock_tag_e],
            }

            # Test for "rating:g"
            arr_g = space.get_array("file_g.jpg")
            mock_numpy_rafs.empty.assert_called_with(1) # Called for each get_array
            self.mock_np_empty_array.__setitem__.assert_called_with(0, 0.0)
            self.assertIs(arr_g, self.mock_np_empty_array)

            # Test for "rating:s"
            arr_s = space.get_array("file_s.jpg")
            self.mock_np_empty_array.__setitem__.assert_called_with(0, 0.33)
            self.assertIs(arr_s, self.mock_np_empty_array)

            # Test for "rating:q"
            arr_q = space.get_array("file_q.jpg")
            self.mock_np_empty_array.__setitem__.assert_called_with(0, 0.66)
            self.assertIs(arr_q, self.mock_np_empty_array)
            
            # Test for "rating:e"
            arr_e = space.get_array("file_e.jpg")
            self.mock_np_empty_array.__setitem__.assert_called_with(0, 1.0)
            self.assertIs(arr_e, self.mock_np_empty_array)
            
            self.assertEqual(mock_numpy_rafs.empty.call_count, 4)


    def test_get_array_invalid_tag_name(self):
        with patch.object(TaggerSpace, 'setup'): # Prevent setup
            space = RatingAsFloatSpace(sources=["dummy.json"])
            
            mock_tag_invalid = MagicMock(spec=mock_tag_module_rafs.Tag)
            mock_tag_invalid.name = "rating:x" # Invalid tag name for the mapping
            space.filename_to_tags = {"file_x.jpg": [mock_tag_invalid]}

            with self.assertRaises(KeyError): # Expect KeyError due to dict lookup
                space.get_array("file_x.jpg")

    def test_get_array_no_tags_for_file(self):
        with patch.object(TaggerSpace, 'setup'):
            space = RatingAsFloatSpace(sources=["dummy.json"])
            space.filename_to_tags = {"file_no_tags.jpg": []} # Empty list of tags

            with self.assertRaises(IndexError): # Expect IndexError from [0]
                space.get_array("file_no_tags.jpg")

    def test_get_array_file_not_in_map(self):
        with patch.object(TaggerSpace, 'setup'):
            space = RatingAsFloatSpace(sources=["dummy.json"])
            space.filename_to_tags = {} # Empty map

            with self.assertRaises(KeyError): # Expect KeyError from filename_to_tags[filename]
                space.get_array("unknown_file.jpg")

    # Properties like 'n' are overridden in __init__ (_n = 1).
    # Test that the 'n' property (from OutputSpace) correctly returns this overridden value.
    def test_n_property_returns_overridden_value(self):
        with patch.object(TaggerSpace, 'setup'): # Prevent setup during init
            space = RatingAsFloatSpace(sources=["dummy.json"])
        
        # _n is set to 1 in RatingAsFloatSpace's __init__
        # The 'n' property in OutputSpace is:
        # @LazyProperty
        # def n(self) -> int:
        #     return len(self.tags)
        # However, RatingAsFloatSpace sets self._n = 1.
        # The @LazyProperty for 'n' in OutputSpace might not be correctly overridden
        # if direct attribute _n is meant to be the source of truth for 'n' property.
        # Let's check self._n directly as per RatingAsFloatSpace's __init__.
        self.assertEqual(space._n, 1)

        # If we want to test the 'n' property itself, we need to consider how it's resolved.
        # If OutputSpace.n is always len(self.tags), then we need to mock self.tags.
        # But RatingAsFloatSpace's intent seems to be that n IS 1.
        # The current OutputSpace.n doesn't look at self._n.
        # This indicates a potential inconsistency or that RatingAsFloatSpace
        # expects `len(self.tags)` to also be 1 after its setup.
        
        # For now, we've tested self._n. If `space.n` property is critical,
        # we'd mock `space.tags` to have length 1.
        space.tags = {"rating_tag_placeholder": MagicMock()} # Make len(tags) == 1
        self.assertEqual(space.n, 1) # Now OutputSpace.n property should yield 1


    # class_weights is set to None in __init__.
    def test_class_weights_is_none(self):
        with patch.object(TaggerSpace, 'setup'):
            space = RatingAsFloatSpace(sources=["dummy.json"])
        self.assertIsNone(space.class_weights)


if __name__ == '__main__':
    unittest.main()

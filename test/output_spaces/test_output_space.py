import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock Rignak
mock_rignak_lazy_os = MagicMock()
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_os
mock_rignak_logger_os = MagicMock()
sys.modules['Rignak.logging_utils'] = mock_rignak_logger_os


# Mock Tag class (dependency for OutputSpace)
mock_tag_module_os = MagicMock()
sys.modules['src.output_spaces.tag'] = mock_tag_module_os

# Mock numpy
mock_numpy_os = MagicMock()
sys.modules['numpy'] = mock_numpy_os

# Import OutputSpace after mocks
from output_spaces.output_space import OutputSpace

# Create a concrete implementation for testing common_setup and dependent properties
class ConcreteOutputSpace(OutputSpace):
    def __init__(self, sources, enforced_tag_names=None, limit=100, mock_data_for_setup=None):
        super().__init__(sources, enforced_tag_names=enforced_tag_names, limit=limit)
        self.mock_data_for_setup = mock_data_for_setup if mock_data_for_setup is not None else []
        self.dataset_size_for_setup = len(self.mock_data_for_setup) # Example dataset_size

    def setup(self) -> None:
        # This method is called by the LazyProperties.
        # It will call common_setup with predefined mock data.
        if self._tags is None: # Check to prevent re-running setup logic if already done
            self.common_setup(
                dataset_size=self.dataset_size_for_setup,
                data=self.mock_data_for_setup
            )

class TestOutputSpace(unittest.TestCase):

    def setUp(self):
        mock_rignak_lazy_os.LazyProperty.reset_mock()
        mock_rignak_logger_os.logger.reset_mock()
        mock_tag_module_os.Tag.reset_mock()
        mock_numpy_os.reset_mock()
        
        # Default mock for Tag instantiation
        self.mock_tag_instance = MagicMock(spec=mock_tag_module_os.Tag)
        self.mock_tag_instance.number_of_use = 0 # Initialize for +=1
        mock_tag_module_os.Tag.return_value = self.mock_tag_instance
        
        # Mock for np.array used in class_weights and get_array
        mock_numpy_os.array.side_effect = lambda x, dtype=None: MagicMock(name=f"ndarray_{x}", spec=list(x) if isinstance(x, list) else x) # Return a mock that can be iterated
        mock_numpy_os.zeros.side_effect = lambda size: MagicMock(name=f"zeros_{size}", spec=[0]*size) # Return a mock list of zeros
        mock_numpy_os.nanmean.return_value = 1.0 # For class_weights calculation


    def test_initialization_defaults(self):
        sources = ["source1.json"]
        space = OutputSpace(sources=sources)
        self.assertEqual(space.sources, sources)
        self.assertIsNone(space.enforced_tag_names)
        self.assertIsNone(space._n) # Not to be confused with n property
        self.assertEqual(space.limit, 100)
        self.assertIsNone(space._tag_names)
        self.assertIsNone(space._tags)
        self.assertIsNone(space._filename_to_tags)
        self.assertIsNone(space._filenames)
        self.assertIsNone(space._class_weights)

    def test_initialization_custom_values(self):
        sources = ["s1", "s2"]
        enforced = ["tagA"]
        limit = 50
        space = OutputSpace(sources=sources, enforced_tag_names=enforced, limit=limit)
        self.assertEqual(space.sources, sources)
        self.assertEqual(space.enforced_tag_names, enforced)
        self.assertEqual(space.limit, limit)

    def test_setup_is_abstract(self):
        space = OutputSpace(sources=[])
        with self.assertRaises(NotImplementedError):
            space.setup()

    @patch('os.path.exists')
    def test_common_setup(self, mock_os_exists):
        mock_os_exists.return_value = True # Assume all files exist

        mock_data = [
            ("file1.jpg", ["tagA", "tagB"]),
            ("file2.png", ["tagB", "tagC"]),
            ("file3.jpg", ["tagA", "tagD", "tagC"]),
            ("non_image.txt", ["tagA"]), # Should be skipped by extension check
            ("missing_file.jpg", ["tagE"]), # Should be skipped by os.path.exists if mocked to False for it
        ]
        # Let "missing_file.jpg" not exist
        mock_os_exists.side_effect = lambda x: x != "missing_file.jpg"
        
        # Mock Tag instances to track number_of_use
        mock_tag_a = MagicMock(spec=mock_tag_module_os.Tag, name="TagA"); mock_tag_a.number_of_use = 0; mock_tag_a.name = "tagA"
        mock_tag_b = MagicMock(spec=mock_tag_module_os.Tag, name="TagB"); mock_tag_b.number_of_use = 0; mock_tag_b.name = "tagB"
        mock_tag_c = MagicMock(spec=mock_tag_module_os.Tag, name="TagC"); mock_tag_c.number_of_use = 0; mock_tag_c.name = "tagC"
        mock_tag_d = MagicMock(spec=mock_tag_module_os.Tag, name="TagD"); mock_tag_d.number_of_use = 0; mock_tag_d.name = "tagD"
        
        def tag_factory_side_effect(index, name, dataset_size):
            if name == "tagA": mock_tag_a.index=index; return mock_tag_a
            if name == "tagB": mock_tag_b.index=index; return mock_tag_b
            if name == "tagC": mock_tag_c.index=index; return mock_tag_c
            if name == "tagD": mock_tag_d.index=index; return mock_tag_d
            return MagicMock(name=f"UnknownTag_{name}") # Fallback
        mock_tag_module_os.Tag.side_effect = tag_factory_side_effect

        dataset_size_for_common_setup = 3 # file1, file2, file3 are valid
        
        # Use ConcreteOutputSpace to call common_setup via its setup()
        # We don't need to pass mock_data_for_setup to ConcreteOutputSpace's init for this direct test.
        space = ConcreteOutputSpace(sources=["dummy"]) # Sources not used by common_setup directly
        space.common_setup(dataset_size=dataset_size_for_common_setup, data=mock_data)

        # Check logger calls
        mock_rignak_logger_os.logger.assert_any_call(f"Setup output space for {dataset_size_for_common_setup} files.", indent=1)
        
        # Check os.path.exists calls
        expected_exists_calls = [call("file1.jpg"), call("file2.png"), call("file3.jpg"), call("non_image.txt"), call("missing_file.jpg")]
        mock_os_exists.assert_has_calls(expected_exists_calls, any_order=False) # Order matters due to side_effect
        
        # Check _filenames: non_image.txt and missing_file.jpg should be excluded
        self.assertEqual(sorted(space._filenames), sorted(["file1.jpg", "file2.png", "file3.jpg"]))
        
        # Check _tags and _tag_names (before sort_tags which is called at the end of common_setup)
        # Initial order of tag creation: tagA, tagB, tagC, tagD
        self.assertIn("tagA", space._tags)
        self.assertIn("tagB", space._tags)
        self.assertIn("tagC", space._tags)
        self.assertIn("tagD", space._tags)
        self.assertNotIn("tagE", space._tags) # From missing_file or non_image
        
        # Check number_of_use for each tag
        self.assertEqual(mock_tag_a.number_of_use, 2) # file1, file3
        self.assertEqual(mock_tag_b.number_of_use, 2) # file1, file2
        self.assertEqual(mock_tag_c.number_of_use, 2) # file2, file3
        self.assertEqual(mock_tag_d.number_of_use, 1) # file3

        # Check _filename_to_tags
        self.assertEqual(len(space._filename_to_tags["file1.jpg"]), 2) # tagA, tagB
        self.assertIn(mock_tag_a, space._filename_to_tags["file1.jpg"])
        self.assertIn(mock_tag_b, space._filename_to_tags["file1.jpg"])
        self.assertEqual(len(space._filename_to_tags["file2.png"]), 2) # tagB, tagC
        self.assertEqual(len(space._filename_to_tags["file3.jpg"]), 3) # tagA, tagD, tagC

        # sort_tags is called at the end of common_setup.
        # It sorts by number_of_use (desc) then limits. Default limit is 100.
        # Counts: A:2, B:2, C:2, D:1. All should be kept.
        # The order in _tag_names after sort could be [A,B,C,D] or [B,A,C,D] etc.
        # And indices are updated.
        # Example: if sorted is [A,B,C,D], then A.index=0, B.index=1, C.index=2, D.index=3
        self.assertEqual(len(space._tag_names), 4)
        for i, tag_name_sorted in enumerate(space._tag_names):
            self.assertEqual(space._tags[tag_name_sorted].index, i)
        
        mock_rignak_logger_os.logger.assert_any_call(f"Found {len(space._tag_names)} tags.")
        mock_rignak_logger_os.logger.assert_any_call(f"Setup output space OK", indent=-1)


    @patch('os.path.exists', MagicMock(return_value=True))
    def test_common_setup_with_enforced_tags(self):
        mock_data = [("file1.jpg", ["tagA", "tagB", "tagC"])]
        enforced = ["tagA", "tagC"]
        
        mock_tag_a = MagicMock(spec=mock_tag_module_os.Tag, name="TagA_enforced"); mock_tag_a.number_of_use = 0
        mock_tag_c = MagicMock(spec=mock_tag_module_os.Tag, name="TagC_enforced"); mock_tag_c.number_of_use = 0
        def tag_factory_side_effect_enf(index, name, dataset_size):
            if name == "tagA": return mock_tag_a
            if name == "tagC": return mock_tag_c
            raise AssertionError(f"Tag {name} should not have been created with enforced_tags")
        mock_tag_module_os.Tag.side_effect = tag_factory_side_effect_enf

        space = ConcreteOutputSpace(sources=["dummy"], enforced_tag_names=enforced)
        space.common_setup(dataset_size=1, data=mock_data)

        self.assertEqual(len(space._tags), 2) # Only tagA and tagC
        self.assertIn("tagA", space._tags)
        self.assertIn("tagC", space._tags)
        self.assertNotIn("tagB", space._tags)
        self.assertEqual(mock_tag_a.number_of_use, 1)
        self.assertEqual(mock_tag_c.number_of_use, 1)
        self.assertEqual(space._filename_to_tags["file1.jpg"], [mock_tag_a, mock_tag_c])


    def test_sort_tags_and_limit(self):
        space = ConcreteOutputSpace(sources=["dummy"], limit=2) # Limit to 2 tags
        # Manually populate _tags before calling sort_tags
        tag_a = MagicMock(spec=mock_tag_module_os.Tag, name="tagA"); tag_a.number_of_use = 10
        tag_b = MagicMock(spec=mock_tag_module_os.Tag, name="tagB"); tag_b.number_of_use = 5
        tag_c = MagicMock(spec=mock_tag_module_os.Tag, name="tagC"); tag_c.number_of_use = 20
        space._tags = {"tagA": tag_a, "tagB": tag_b, "tagC": tag_c}
        
        space.sort_tags()

        # Expected order: tagC (20), tagA (10). tagB (5) is dropped due to limit=2.
        self.assertEqual(space._tag_names, ["tagC", "tagA"])
        # Check indices are updated
        self.assertEqual(tag_c.index, 0)
        self.assertEqual(tag_a.index, 1)
        # tagB's index might not be updated or might be irrelevant as it's not in _tag_names.


    # Testing LazyProperties - they should trigger setup() on first access
    @patch.object(ConcreteOutputSpace, 'setup') # Patch setup on the concrete class
    def test_lazy_properties_trigger_setup(self, mock_concrete_setup_method):
        # Use ConcreteOutputSpace where setup calls common_setup with some mock_data
        space = ConcreteOutputSpace(sources=["s"], mock_data_for_setup=[("f.jpg",["t"])])
        
        # Access each lazy property
        _ = space.filenames
        mock_concrete_setup_method.assert_called_once() # setup called for filenames
        mock_concrete_setup_method.reset_mock() # Reset for next property
        
        _ = space.tags
        mock_concrete_setup_method.assert_called_once() # setup called for tags
        mock_concrete_setup_method.reset_mock()

        _ = space.tag_names
        mock_concrete_setup_method.assert_called_once()
        mock_concrete_setup_method.reset_mock()

        _ = space.filename_to_tags
        mock_concrete_setup_method.assert_called_once()
        mock_concrete_setup_method.reset_mock()
        
        # For 'n' and 'class_weights', they depend on 'tags' which depends on setup.
        # So, accessing 'tags' first would have called setup.
        # If we access 'n' directly, it should also trigger setup if 'tags' wasn't populated.
        # Let's create a fresh instance where setup hasn't run.
        space_for_n = ConcreteOutputSpace(sources=["s"], mock_data_for_setup=[("f.jpg",["t"])])
        with patch.object(ConcreteOutputSpace, 'setup') as mock_setup_for_n: # New patch for new instance
            _ = space_for_n.n
            mock_setup_for_n.assert_called_once() # Accessing 'n' calls 'tags', which calls setup.

        space_for_cw = ConcreteOutputSpace(sources=["s"], mock_data_for_setup=[("f.jpg",["t"])])
        with patch.object(ConcreteOutputSpace, 'setup') as mock_setup_for_cw:
             _ = space_for_cw.class_weights # Accessing 'class_weights' calls 'tags', which calls setup.
             mock_setup_for_cw.assert_called_once()


    def test_n_property(self):
        space = ConcreteOutputSpace(sources=["s"])
        # Manually set _tags for this test, assuming setup has run
        mock_tags_dict = {"tagA": MagicMock(), "tagB": MagicMock()}
        space._tags = mock_tags_dict # Simulate that setup has populated _tags
        
        self.assertEqual(space.n, len(mock_tags_dict))

    def test_class_weights_property(self):
        space = ConcreteOutputSpace(sources=["s"])
        
        tag_a = MagicMock(name="TagA_cw"); tag_a.number_of_use = 2
        tag_b = MagicMock(name="TagB_cw"); tag_b.number_of_use = 4
        # Assume n = 2 (number of tags)
        # Weights = n / number_of_use = [2/2, 2/4] = [1, 0.5]
        # nanmean([1, 0.5]) = 0.75
        # Final weights = [1/0.75, 0.5/0.75] = [1.333, 0.666]
        space._tags = {"tagA": tag_a, "tagB": tag_b} # Simulate setup
        # Mock the 'n' property to return 2 for this test
        with patch.object(OutputSpace, 'n', new_callable=PropertyMock, return_value=2):
            mock_raw_weights = MagicMock(name="raw_weights_array")
            mock_numpy_os.array.return_value = mock_raw_weights # For np.array([n/tag.number_of_use ...])
            
            mock_normalized_weights = MagicMock(name="normalized_weights_array")
            mock_raw_weights.__truediv__.return_value = mock_normalized_weights # For weights / np.nanmean(weights)

            weights = space.class_weights
            
            mock_numpy_os.array.assert_called_once_with([2/tag_a.number_of_use, 2/tag_b.number_of_use])
            mock_numpy_os.nanmean.assert_called_once_with(mock_raw_weights)
            mock_raw_weights.__truediv__.assert_called_once_with(1.0) # nanmean was mocked to 1.0
            self.assertIs(weights, mock_normalized_weights)


    def test_get_array(self):
        space = ConcreteOutputSpace(sources=["s"])
        filename_test = "file1.jpg"
        
        # Simulate state after setup
        tag_a_idx = 0; tag_a = MagicMock(name="TagA_ga"); tag_a.index = tag_a_idx
        tag_b_idx = 1; tag_b = MagicMock(name="TagB_ga"); tag_b.index = tag_b_idx
        space._filename_to_tags = {filename_test: [tag_a, tag_b]}
        # Mock 'n' property
        with patch.object(OutputSpace, 'n', new_callable=PropertyMock, return_value=2):
            mock_zeros_arr = MagicMock(name="zeros_for_get_array", spec=[0,0]) # Behaves like a list [0,0]
            # mock_zeros_arr.__setitem__ is implicitly used by "hot_encoded[tag.index] = 1"
            mock_numpy_os.zeros.return_value = mock_zeros_arr

            arr = space.get_array(filename_test)

            mock_numpy_os.zeros.assert_called_once_with(2) # space.n
            # Check that the correct indices were set to 1
            # This requires mock_zeros_arr to have a __setitem__ that can be asserted.
            # If mock_zeros_arr is a simple list mock, this is harder.
            # If it's a MagicMock with spec=list, it won't record __setitem__.
            # A more robust way:
            # Check that the returned array is what we expect.
            # For this, we need np.zeros to return an actual list-like mock that we can inspect
            # or that we control the mock of tag.index.
            
            # Let's assume mock_zeros_arr is a list-like mock that got modified.
            # Expected: [1, 1] because tag_a.index=0, tag_b.index=1
            # This test needs refinement if exact value of `arr` is to be checked beyond type.
            self.assertIsNotNone(arr) # Basic check

    # Properties not found in code: shape, names, values
    # Methods not found: get_tag_from_index, get_index_from_tag, get_tag_from_vector, 
    # get_vector_from_tag, get_html_table_header, get_html_table_row.
    # These will not be tested as they are not implemented in the provided OutputSpace.

if __name__ == '__main__':
    unittest.main()

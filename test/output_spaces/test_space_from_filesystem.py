import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock base OutputSpace and its dependencies
mock_rignak_lazy_fs = MagicMock()
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_fs
mock_rignak_logger_fs = MagicMock()
sys.modules['Rignak.logging_utils'] = mock_rignak_logger_fs
mock_tag_module_fs = MagicMock()
sys.modules['src.output_spaces.tag'] = mock_tag_module_fs
mock_numpy_fs = MagicMock() # OutputSpace uses numpy
sys.modules['numpy'] = mock_numpy_fs

# Import CategorizationSpace after mocks
from output_spaces.space_from_filesystem import CategorizationSpace
from output_spaces.output_space import OutputSpace # For patching common_setup

class TestCategorizationSpace(unittest.TestCase):

    def setUp(self):
        mock_rignak_lazy_fs.LazyProperty.reset_mock()
        mock_rignak_logger_fs.logger.reset_mock()
        mock_tag_module_fs.Tag.reset_mock()
        mock_numpy_fs.reset_mock()

    def test_initialization(self):
        sources_val = ["dir1/file1.jpg", "dir2/file2.png"]
        enforced_tags_val = ["tagX"]
        limit_val = 20
        
        space = CategorizationSpace(
            sources=sources_val, 
            enforced_tag_names=enforced_tags_val, 
            limit=limit_val
        )
        
        self.assertEqual(space.sources, sources_val)
        self.assertEqual(space.enforced_tag_names, enforced_tags_val)
        self.assertEqual(space.limit, limit_val)
        self.assertIsNone(space._tags) # Lazy properties from parent not yet set


    @patch('os.path.dirname')
    @patch('os.path.basename')
    @patch.object(OutputSpace, 'common_setup') # Mock common_setup from the parent
    def test_setup_method(self, mock_common_setup, mock_os_basename, mock_os_dirname):
        sources_val = [
            "path/to/classA/img1.jpg",
            "path/to/classB/img2.png",
            "another/path/classA/img3.jpg"
        ]
        
        # Mock os.path.dirname and os.path.basename
        # dirname("path/to/classA/img1.jpg") -> "path/to/classA"
        # basename("path/to/classA") -> "classA"
        def dirname_side_effect(path):
            if path == sources_val[0]: return "path/to/classA"
            if path == sources_val[1]: return "path/to/classB"
            if path == sources_val[2]: return "another/path/classA"
            return ""
        mock_os_dirname.side_effect = dirname_side_effect
        
        def basename_side_effect(path):
            if path == "path/to/classA": return "classA"
            if path == "path/to/classB": return "classB"
            if path == "another/path/classA": return "classA" # Can be same class name from different path
            return ""
        mock_os_basename.side_effect = basename_side_effect
            
        space = CategorizationSpace(sources=sources_val)
        space.setup() # Call directly for testing

        # 1. Check os.path.dirname and os.path.basename calls
        expected_dirname_calls = [call(s) for s in sources_val]
        mock_os_dirname.assert_has_calls(expected_dirname_calls, any_order=False)
        
        expected_basename_calls = [
            call("path/to/classA"), 
            call("path/to/classB"), 
            call("another/path/classA")
        ]
        mock_os_basename.assert_has_calls(expected_basename_calls, any_order=False)
        
        # 2. Check common_setup call
        expected_dataset_size = len(sources_val) # 3 in this case
        expected_data_for_common_setup = [
            (sources_val[0], ["classA"]),
            (sources_val[1], ["classB"]),
            (sources_val[2], ["classA"])
        ]
        mock_common_setup.assert_called_once_with(expected_dataset_size, expected_data_for_common_setup)


    @patch.object(CategorizationSpace, 'setup') # Patch CategorizationSpace's own setup
    def test_lazy_properties_trigger_categorizationspace_setup(self, mock_categorizationspace_setup_method):
        sources_val = ["dir/f.jpg"]
        space = CategorizationSpace(sources=sources_val)
        
        _ = space.tag_names # Access a lazy property that depends on setup
        
        mock_categorizationspace_setup_method.assert_called_once()

    # As with TaggerSpace, other properties and methods are inherited from OutputSpace
    # and tested in test_output_space.py. The key for CategorizationSpace is that its
    # `setup` method correctly derives tags from directory structure and calls `common_setup`.

if __name__ == '__main__':
    unittest.main()

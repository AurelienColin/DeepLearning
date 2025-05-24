import unittest
from unittest.mock import MagicMock, patch, mock_open, call
import sys
import os
import json # Import json for json.load mocking

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock base OutputSpace and its dependencies if TaggerSpace uses them directly
# Rignak and Tag are used by OutputSpace, which TaggerSpace inherits from.
mock_rignak_lazy_ts = MagicMock()
sys.modules['Rignak.lazy_property'] = mock_rignak_lazy_ts
mock_rignak_logger_ts = MagicMock()
sys.modules['Rignak.logging_utils'] = mock_rignak_logger_ts
mock_tag_module_ts = MagicMock()
sys.modules['src.output_spaces.tag'] = mock_tag_module_ts
mock_numpy_ts = MagicMock() # OutputSpace uses numpy
sys.modules['numpy'] = mock_numpy_ts


# Import TaggerSpace after mocks
from output_spaces.space_from_json import TaggerSpace
from output_spaces.output_space import OutputSpace # For patching common_setup

class TestTaggerSpace(unittest.TestCase):

    def setUp(self):
        mock_rignak_lazy_ts.LazyProperty.reset_mock()
        mock_rignak_logger_ts.logger.reset_mock()
        mock_tag_module_ts.Tag.reset_mock()
        mock_numpy_ts.reset_mock()

    def test_initialization(self):
        sources_val = ["path/to/data.json"]
        enforced_tags_val = ["tag1", "tag2"]
        limit_val = 50
        
        space = TaggerSpace(
            sources=sources_val, 
            enforced_tag_names=enforced_tags_val, 
            limit=limit_val
        )
        
        self.assertEqual(space.sources, sources_val)
        self.assertEqual(space.enforced_tag_names, enforced_tags_val)
        self.assertEqual(space.limit, limit_val)
        # Lazy properties from parent are not yet set
        self.assertIsNone(space._tags)


    @patch('os.path.dirname')
    @patch('builtins.open', new_callable=mock_open) # Mocks open()
    @patch('json.load')
    @patch.object(OutputSpace, 'common_setup') # Mock common_setup from the parent
    def test_setup_method(self, mock_common_setup, mock_json_load, mock_file_open, mock_os_dirname):
        json_file_path = "dummy_dir/data.json"
        sources_val = [json_file_path]
        
        # Mock os.path.dirname to return a specific directory
        mock_os_dirname.return_value = "dummy_dir"
        
        # Mock data that json.load will return
        mock_json_data_content = {
            "image1.jpg": ["tagA", "tagB"],
            "image2.png": ["tagB", "tagC"]
        }
        mock_json_load.return_value = mock_json_data_content
        
        # Instantiate TaggerSpace
        space = TaggerSpace(sources=sources_val)
        
        # Call setup (which is usually called by lazy properties)
        space.setup()

        # 1. Check os.path.dirname call
        mock_os_dirname.assert_called_once_with(json_file_path)
        
        # 2. Check file open call
        mock_file_open.assert_called_once_with(json_file_path, 'r')
        
        # 3. Check json.load call
        # The first argument to json.load is the file handle from open()
        mock_json_load.assert_called_once_with(mock_file_open.return_value)
        
        # 4. Check common_setup call
        expected_dataset_size = len(mock_json_data_content) # 2 in this case
        expected_data_for_common_setup = [
            (os.path.join("dummy_dir", "image1.jpg"), ["tagA", "tagB"]),
            (os.path.join("dummy_dir", "image2.png"), ["tagB", "tagC"])
        ]
        mock_common_setup.assert_called_once_with(expected_dataset_size, expected_data_for_common_setup)

    # Test that lazy properties from OutputSpace (like filenames, tags, etc.)
    # correctly trigger TaggerSpace's setup method.
    @patch.object(TaggerSpace, 'setup') # Patch TaggerSpace's own setup
    def test_lazy_properties_trigger_taggerspace_setup(self, mock_taggerspace_setup_method):
        sources_val = ["path/to/some.json"]
        space = TaggerSpace(sources=sources_val)
        
        # Access a lazy property that depends on setup
        _ = space.filenames 
        
        mock_taggerspace_setup_method.assert_called_once()

    # Other properties and methods like get_array, class_weights, n, tag_names, etc.,
    # are inherited from OutputSpace and their core logic (once setup is done)
    # is tested in test_output_space.py.
    # The key for TaggerSpace is that its `setup` method correctly parses the JSON
    # and provides the right arguments to `common_setup`.

if __name__ == '__main__':
    unittest.main()

import json
import os
import shutil
import tempfile
import pytest # Added for pytest.raises
from typing import Any, Dict, List, Set

import numpy as np

# Import Tag from its specific module
from src.output_spaces.tag import Tag
# Import OutputSpace and TaggerSpace
from src.output_spaces.output_space import OutputSpace
from src.output_spaces.space_from_json import TaggerSpace


class TestOutputSpaceBase:
    """
    Base class for testing output space functionalities.
    Handles the setup and teardown of dummy files and directories needed for tests.
    """
    tmpdir: str
    dummy_data_path: str
    annotations_filename: str
    dummy_json_content: Dict[str, List[str]]
    image_files_to_create: List[str]
    # This is the total number of entries in the JSON, used for Tag.dataset_size
    expected_raw_dataset_size: int

    def setup_method(self) -> None:
        """
        Set up a temporary directory with dummy annotation data and image files.
        """
        self.tmpdir = tempfile.mkdtemp()
        self.dummy_data_path = os.path.join(self.tmpdir, "dummy_data")
        os.makedirs(self.dummy_data_path, exist_ok=True)

        self.annotations_filename = os.path.join(self.dummy_data_path, "annotations.json")

        self.dummy_json_content = {
            "image1.png": ["tagA", "tagB"],
            "image2.jpg": ["tagB", "tagC"],
            "image3.png": ["tagA"],
            "image4.bmp": ["tagD"],  # This file won't be created, to test handling of missing files
            "image5.png": ["tagA", "tagB", "tagC"],
        }
        # Tag.dataset_size is based on the raw number of entries in the JSON
        self.expected_raw_dataset_size = len(self.dummy_json_content) # Should be 5

        with open(self.annotations_filename, 'w') as f:
            json.dump(self.dummy_json_content, f)

        # These are the files that should actually be loaded by TaggerSpace
        # (exist and have valid extensions)
        self.image_files_to_create = ["image1.png", "image2.jpg", "image3.png", "image5.png"]
        for filename in self.image_files_to_create:
            # Create empty files
            with open(os.path.join(self.dummy_data_path, filename), 'w') as f:
                pass # Create an empty file

    def teardown_method(self) -> None:
        """
        Remove the temporary directory and its contents.
        """
        shutil.rmtree(self.tmpdir)


@pytest.fixture
def tagger_space_setup():
    # Create an instance of the base class to use its setup/teardown
    base_setup = TestOutputSpaceBase()
    base_setup.setup_method()
    yield base_setup # This is what the test function will receive
    base_setup.teardown_method()


class TestTaggerSpace:

    def test_sources_after_setup(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests the 'sources' property of TaggerSpace after setup."""
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename])
        _ = ts.tag_names  # Force setup
        assert ts.sources == [tagger_space_setup.annotations_filename]

    def test_filenames_after_setup(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests the 'filenames' property of TaggerSpace after setup."""
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename])
        _ = ts.tag_names  # Force setup
        assert len(ts.filenames) == len(tagger_space_setup.image_files_to_create)
        expected_abs_filenames: Set[str] = {
            os.path.join(tagger_space_setup.dummy_data_path, f) for f in tagger_space_setup.image_files_to_create
        }
        assert set(ts.filenames) == expected_abs_filenames

    def test_tag_names_and_count_after_setup(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests 'tag_names' and the count of tags after setup."""
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename])
        _ = ts.tag_names  # Force setup
        expected_tag_names_ordered: List[str] = ["tagA", "tagB", "tagC"]
        assert ts.tag_names == expected_tag_names_ordered
        assert len(ts.tag_names) == 3
        assert len(ts.tags) == 3

    def test_tag_object_properties_after_setup(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests properties of individual Tag objects within TaggerSpace after setup."""
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename])
        _ = ts.tag_names  # Force setup
        expected_tag_names_ordered: List[str] = ["tagA", "tagB", "tagC"]
        tag_counts: Dict[str, int] = {"tagA": 3, "tagB": 3, "tagC": 2}
        for i, tag_name in enumerate(expected_tag_names_ordered):
            assert tag_name in ts.tags
            tag_object: Tag = ts.tags[tag_name]
            assert tag_object.name == tag_name
            assert tag_object.index == i
            assert tag_object.number_of_use == tag_counts[tag_name]
            assert tag_object.dataset_size == tagger_space_setup.expected_raw_dataset_size

    def test_n_value_after_setup(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests the 'n' property (number of unique tags) of TaggerSpace after setup."""
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename])
        _ = ts.tag_names  # Force setup
        assert ts.n == 3

    def test_filename_to_tags_structure_and_content_after_setup(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests the 'filename_to_tags' mapping after TaggerSpace setup."""
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename])
        _ = ts.tag_names  # Force setup
        assert len(ts.filename_to_tags) == len(tagger_space_setup.image_files_to_create)

        f1_path: str = os.path.join(tagger_space_setup.dummy_data_path, "image1.png")
        assert f1_path in ts.filename_to_tags
        assert sorted([t.name for t in ts.filename_to_tags[f1_path]]) == sorted(["tagA", "tagB"])

        f2_path: str = os.path.join(tagger_space_setup.dummy_data_path, "image2.jpg")
        assert f2_path in ts.filename_to_tags
        assert sorted([t.name for t in ts.filename_to_tags[f2_path]]) == sorted(["tagB", "tagC"])

        f3_path: str = os.path.join(tagger_space_setup.dummy_data_path, "image3.png")
        assert f3_path in ts.filename_to_tags
        assert sorted([t.name for t in ts.filename_to_tags[f3_path]]) == sorted(["tagA"])

        f5_path: str = os.path.join(tagger_space_setup.dummy_data_path, "image5.png")
        assert f5_path in ts.filename_to_tags
        assert sorted([t.name for t in ts.filename_to_tags[f5_path]]) == sorted(["tagA", "tagB", "tagC"])

    def test_get_array_for_file_with_subset_of_tags(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests get_array for a file associated with a subset of all known tags."""
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename])
        _ = ts.tag_names  # Force setup

        f1_path: str = os.path.join(tagger_space_setup.dummy_data_path, "image1.png")
        arr1: np.ndarray = ts.get_array(f1_path)
        
        expected_arr1: np.ndarray = np.zeros(ts.n, dtype=float)
        expected_arr1[ts.tags["tagA"].index] = 1.0
        expected_arr1[ts.tags["tagB"].index] = 1.0
        np.testing.assert_array_equal(arr1, expected_arr1)

    def test_get_array_for_file_with_all_tags(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests get_array for a file associated with all known tags."""
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename])
        _ = ts.tag_names  # Force setup

        f5_path: str = os.path.join(tagger_space_setup.dummy_data_path, "image5.png")
        arr5: np.ndarray = ts.get_array(f5_path)

        expected_arr5: np.ndarray = np.zeros(ts.n, dtype=float)
        expected_arr5[ts.tags["tagA"].index] = 1.0
        expected_arr5[ts.tags["tagB"].index] = 1.0
        expected_arr5[ts.tags["tagC"].index] = 1.0
        np.testing.assert_array_equal(arr5, expected_arr5)
        # For image5, all tags are present, so the array should be all ones.
        np.testing.assert_array_equal(arr5, np.ones(ts.n, dtype=float))

    def test_class_weights(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename])
        _ = ts.tag_names

        weights: np.ndarray = ts.class_weights
        assert weights.shape == (ts.n,)

        uses: np.ndarray = np.array([ts.tags[name].number_of_use for name in ts.tag_names])
        raw_weights_from_impl: np.ndarray = np.array([ts.n / ts.tags[name].number_of_use for name in ts.tag_names])
        expected_weights: np.ndarray = raw_weights_from_impl / np.mean(raw_weights_from_impl)
        
        np.testing.assert_array_almost_equal(weights, expected_weights)

    def test_setup_with_enforced_tag_names(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests TaggerSpace setup properties when enforced_tag_names is used."""
        enforced_tags_list: List[str] = ["tagA", "tagC"]
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename], enforced_tag_names=enforced_tags_list)
        _ = ts.tag_names  # Force setup

        assert ts.tag_names == enforced_tags_list
        assert ts.n == len(enforced_tags_list)
        
        assert ts.tags["tagA"].number_of_use == 3
        assert ts.tags["tagC"].number_of_use == 2
        assert ts.tags["tagA"].dataset_size == tagger_space_setup.expected_raw_dataset_size
        assert ts.tags["tagC"].dataset_size == tagger_space_setup.expected_raw_dataset_size
        assert "tagB" not in ts.tags

    def test_get_array_with_enforced_tags_file1(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests get_array with enforced_tag_names for image1.png."""
        enforced_tags_list: List[str] = ["tagA", "tagC"]
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename], enforced_tag_names=enforced_tags_list)
        _ = ts.tag_names  # Force setup

        f1_path: str = os.path.join(tagger_space_setup.dummy_data_path, "image1.png")
        # Original tags for image1.png: ["tagA", "tagB"]. With enforcement ["tagA", "tagC"], only "tagA" should be active.
        arr1: np.ndarray = ts.get_array(f1_path)
        
        expected_arr1: np.ndarray = np.zeros(ts.n, dtype=float)
        expected_arr1[ts.tags["tagA"].index] = 1.0
        np.testing.assert_array_equal(arr1, expected_arr1)

    def test_get_array_with_enforced_tags_file2(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests get_array with enforced_tag_names for image2.jpg."""
        enforced_tags_list: List[str] = ["tagA", "tagC"]
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename], enforced_tag_names=enforced_tags_list)
        _ = ts.tag_names  # Force setup

        f2_path: str = os.path.join(tagger_space_setup.dummy_data_path, "image2.jpg")
        # Original tags for image2.jpg: ["tagB", "tagC"]. With enforcement ["tagA", "tagC"], only "tagC" should be active.
        arr2: np.ndarray = ts.get_array(f2_path)
        
        expected_arr2: np.ndarray = np.zeros(ts.n, dtype=float)
        expected_arr2[ts.tags["tagC"].index] = 1.0
        np.testing.assert_array_equal(arr2, expected_arr2)

    def test_setup_with_tag_limit(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests TaggerSpace setup properties when a tag limit is applied."""
        limit: int = 2
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename], limit=limit)
        _ = ts.tag_names  # Force setup

        assert ts.n == limit
        assert ts.tag_names == ["tagA", "tagB"]
        assert "tagC" not in ts.tags
        assert "tagC" not in ts.tag_names

    def test_get_array_with_tag_limit(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests get_array when a tag limit is applied."""
        limit: int = 2
        ts: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename], limit=limit)
        _ = ts.tag_names  # Force setup

        f5_path: str = os.path.join(tagger_space_setup.dummy_data_path, "image5.png")
        # Original tags for image5.png: ["tagA", "tagB", "tagC"]. 
        # With limit=2, space has ["tagA", "tagB"]. So, for image5.png, "tagA" and "tagB" should be active.
        arr5: np.ndarray = ts.get_array(f5_path)
        
        expected_arr5: np.ndarray = np.zeros(ts.n, dtype=float) # ts.n is 2
        expected_arr5[ts.tags["tagA"].index] = 1.0
        expected_arr5[ts.tags["tagB"].index] = 1.0
        np.testing.assert_array_equal(arr5, expected_arr5)

    # --- New tests start here ---

    def test_init_with_empty_sources(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        ts = TaggerSpace(sources=[])
        assert ts.sources == []
        assert ts.tag_names == []
        assert ts.filenames == []
        assert ts.n == 0
        assert ts.filename_to_tags == {}
        assert ts.tags == {}

    def test_init_with_non_existent_json(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        non_existent_path = os.path.join(tagger_space_setup.dummy_data_path, "non_existent_annotations.json")
        ts = TaggerSpace(sources=[non_existent_path])
        with pytest.raises(FileNotFoundError):
            _ = ts.tag_names # Accessing any lazy property that triggers setup

    def test_setup_with_empty_json(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        empty_json_path = os.path.join(tagger_space_setup.dummy_data_path, "empty.json")
        with open(empty_json_path, 'w') as f:
            json.dump({}, f)
        
        ts = TaggerSpace(sources=[empty_json_path])
        assert ts.tag_names == []
        assert ts.filenames == []
        assert ts.n == 0
        assert ts.tags == {}
        assert ts.filename_to_tags == {}

    def test_setup_with_no_processable_entries_json(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        no_process_json_path = os.path.join(tagger_space_setup.dummy_data_path, "no_process.json")
        # image.txt is not a valid extension, image_not_exist.png does not exist
        content = {
            "image.txt": ["tagA"],
            "image_not_exist.png": ["tagB"] 
        }
        with open(no_process_json_path, 'w') as f:
            json.dump(content, f)
            
        ts = TaggerSpace(sources=[no_process_json_path])
        assert ts.tag_names == []
        assert ts.filenames == []
        assert ts.n == 0
        assert ts.tags == {}
        assert ts.filename_to_tags == {}

    def test_get_array_unknown_filename(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        ts = TaggerSpace(sources=[tagger_space_setup.annotations_filename])
        _ = ts.tag_names # Ensure setup is done
        
        # Create an absolute path for the unknown image, similar to how TaggerSpace stores them
        unknown_image_abs_path = os.path.join(tagger_space_setup.dummy_data_path, "unknown_image.png")
        
        with pytest.raises(KeyError):
            ts.get_array(unknown_image_abs_path)

    def test_enforced_tags_are_entirely_new(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests behavior when enforced_tag_names contains only tags not in the dataset."""
        ts1: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename], enforced_tag_names=["newTagX", "newTagY"])
        _ = ts1.tag_names  # Force setup
        assert ts1.tag_names == []
        assert ts1.n == 0
        assert ts1.tags == {}

    def test_enforced_tags_mix_existing_and_new(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests behavior when enforced_tag_names contains a mix of existing and new tags."""
        ts2: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename], enforced_tag_names=["tagA", "newTagX"])
        _ = ts2.tag_names  # Force setup
        
        # Only "tagA" should be present as "newTagX" is not in the dataset,
        # and "tagB", "tagC" are filtered out because they are not in enforced_tag_names.
        assert ts2.tag_names == ["tagA"]
        assert ts2.n == 1
        assert "tagA" in ts2.tags
        assert "newTagX" not in ts2.tags
        assert ts2.tags["tagA"].number_of_use == 3 # Original count from data
        assert ts2.tags["tagA"].dataset_size == tagger_space_setup.expected_raw_dataset_size

    def test_limit_zero(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests TaggerSpace behavior when the tag limit is set to 0."""
        ts0: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename], limit=0)
        _ = ts0.tag_names  # Force setup
        assert ts0.tag_names == []
        assert ts0.n == 0
        assert ts0.tags == {}

    def test_limit_one(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests TaggerSpace behavior when the tag limit is set to 1."""
        ts1: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename], limit=1)
        _ = ts1.tag_names  # Force setup
        assert ts1.n == 1
        # Expected order: tagA (3 uses), tagB (3 uses), tagC (2 uses). A before B alphabetically.
        assert ts1.tag_names == ["tagA"]
        assert "tagA" in ts1.tags
        assert "tagB" not in ts1.tags
        assert "tagC" not in ts1.tags

    def test_limit_greater_than_available_tags(self, tagger_space_setup: TestOutputSpaceBase) -> None:
        """Tests TaggerSpace behavior when tag limit exceeds the number of unique tags."""
        # There are 3 unique tags (tagA, tagB, tagC) in the dummy data.
        ts10: TaggerSpace = TaggerSpace(sources=[tagger_space_setup.annotations_filename], limit=10)
        _ = ts10.tag_names  # Force setup
        assert ts10.n == 3
        assert ts10.tag_names == ["tagA", "tagB", "tagC"]

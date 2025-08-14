import pytest

from ML.src.output_spaces.custom.sample import Sample, get_samples
import numpy as np
import os
from rignak.src.custom_requests.request_utils import download_file
from ML.src.scripts.utils.download_from_danbooru import JsonDownloader

@pytest.fixture(scope="module")
def sample() -> Sample:
    sample_id = "id:4424469"
    return get_samples(sample_id, 1)[0]


def test_sample_url(sample: Sample) -> None:
    assert sample.url == "https://cdn.donmai.us/original/03/a4/03a4fa48fa126d6bdbdece091a7c0ad3.jpg"


def test_sample_tag(sample: Sample) -> None:
    assert "sonozaki_shion" in sample.tags


def test_sample_array(sample: Sample) -> None:
    expected_outputs = (0, 0, 3, None, 4, 4, None, 1, 2, 5, 2, 0, 0, 3, 3, None, None, None, 2, 0)
    outputs = tuple((np.argmax(e) if e.sum() else None) for e in sample.output)
    assert len(expected_outputs) == len(outputs)
    assert outputs == expected_outputs


def test_download_sample(sample: Sample) -> None:
    filename = f'test/.tmp/{os.path.basename(sample.url)}'
    headers = {'login': JsonDownloader.login, 'api_key': JsonDownloader.api_key}
    try:
        download_file(sample.url, filename, headers=headers)
    except Exception:
        pytest.fail(f"Failed to download {sample.url}")
    else:
        assert os.path.exists(filename)
        os.remove(filename)

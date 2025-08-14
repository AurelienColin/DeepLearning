import os.path
from dataclasses import dataclass
import typing
from rignak.src.logging_utils import logger
from ML.src.output_spaces.custom import hierarchical_tags
from rignak.src.custom_requests.request_utils import download_file
from ML.src.output_spaces.custom.sample import get_samples
import json
from ML.src.scripts.utils.download_from_danbooru import JsonDownloader


@dataclass
class Downloader:
    root_folder: str = ".tmp/dataset/hierarchical"
    entry_per_subcategory: int = 100

    def run(self):
        requests = []
        for category in hierarchical_tags.categories:
            requests += category.get_request_tags()

        urls = []
        samples = []
        logger.set_iterator(len(requests))
        for request in requests:
            request = ' '.join((request, *hierarchical_tags.WHITELIST))
            request = ' -'.join((request, *hierarchical_tags.BLACKLIST))

            current_samples = get_samples(request, self.entry_per_subcategory)
            for sample in current_samples:
                try:
                    assert sample.url is not None
                except AssertionError:
                    continue
                if sample.url not in urls and sample.url.endswith('.jpg'):
                    urls.append(sample.url)
                    samples.append(sample)
            logger.iterate(f"{request} | {len(samples)} samples found.")

        json_data = []
        for i, sample in enumerate(samples):
            json_data.append({
                'filename': f"{self.root_folder}/imgs/{i}{os.path.splitext(sample.url)[1]}",
                'url': sample.url,
                'tags': sample.tags
            })
        filename = f"{self.root_folder}/data.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=4, sort_keys=True)

        logger.set_iterator(len(json_data))

        headers = {'login': JsonDownloader.login, 'api_key': JsonDownloader.api_key}
        logger.set_iterator(len(json_data))
        for entry in json_data:
            download_file(entry['url'], entry['filename'], headers=headers)
            logger.iterate()

if __name__ == "__main__":
    Downloader().run()

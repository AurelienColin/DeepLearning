import os
import typing

import requests
import typing
import json
from rignak.src.logging_utils import logger
from dataclasses import dataclass


@dataclass
class JsonDownloader:
    tags: str
    n_entries: int = 1

    api_endpoint: str = "https://danbooru.donmai.us/posts.json"
    api_key: str = os.environ.get('DANBOORU_KEY')
    login: str = os.environ.get('DANBOORU_LOGIN')

    entries_per_page: int = 100

    def download_single_page(self, page: int) -> typing.List[typing.Dict[str, typing.Any]]:
        params = {
            'login': self.login,
            'api_key': self.api_key,
            'limit': self.entries_per_page,
            'tags': self.tags,
            'page': page,
        }
        try:
            response = requests.get(self.api_endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger(f"Error during request: {e}", level="error")
            return []
        except json.JSONDecodeError as e:
            logger(f"Error decoding JSON response: {e}", level="error")
            return []

    def run(self) -> typing.List[typing.Dict[str, typing.Any]]:
        data = []
        for page in range(int(self.n_entries / self.entries_per_page) + 1):
            data += self.download_single_page(page)
        return data[:self.n_entries]

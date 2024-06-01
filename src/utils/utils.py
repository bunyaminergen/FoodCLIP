# Standard library imports
import os
import tarfile

# Related third party imports
import yaml
import requests
from tqdm import tqdm


class Download:
    def __init__(self, config_path='src/config/datasets.yaml'):
        self.datasets = self.load_datasets(config_path)
        self.ensure_data_directory_exists()

    def load_datasets(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config.get('datasets', {})

    def ensure_data_directory_exists(self):
        data_directory = './.data'
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            print(f"Created directory: {data_directory}")

    def download_and_extract(self, dataset_name):
        if dataset_name not in self.datasets:
            raise ValueError(f"{dataset_name} dataset is unknown. Available datasets: {list(self.datasets.keys())}")

        dataset_info = self.datasets[dataset_name]
        url = dataset_info['url']
        download_path = dataset_info['download_path']
        extract_path = dataset_info['extract_path']

        if not os.path.exists(extract_path) or not os.listdir(extract_path):
            if not os.path.exists(os.path.dirname(extract_path)):
                os.makedirs(os.path.dirname(extract_path))
            print(f"{dataset_name} dataset not found or empty, downloading...")
            self._download(url, download_path)
            self._extract_and_cleanup(download_path, extract_path)
        else:
            print(f"{dataset_name} dataset already exists and is not empty, skipping download.")

    def _download(self, url, download_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(download_path, 'wb') as file, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=download_path) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

    def _extract_and_cleanup(self, download_path, extract_path):
        if tarfile.is_tarfile(download_path):
            with tarfile.open(download_path) as tar:
                tar.extractall(path=os.path.dirname(extract_path))
                extracted_main_dir = os.path.join(os.path.dirname(extract_path), 'food-101')
                if os.path.exists(extracted_main_dir):
                    os.rename(extracted_main_dir, extract_path)
                print(f"Dataset successfully extracted to {extract_path}.")
            os.remove(download_path)
            print(f"Archive file {download_path} deleted.")
        else:
            print("The downloaded file is not a tar archive.")


if __name__ == '__main__':
    downloader = Download(config_path='datasets.yaml')
    downloader.download_and_extract('food-101')

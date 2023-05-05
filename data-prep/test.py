import os
import pytest
from prep import download_and_prepare_data
import shutil
import tempfile
class TestClass:
    def test_download_and_prepare_data_success(self):
        download_and_prepare_data()
        assert os.path.exists("data/train_set.pt")
        assert os.path.exists("data/test_set.pt")

    def test_download_and_prepare_data_directory_creation(self):
        if os.path.exists("data"):
            shutil.rmtree("data")
        download_and_prepare_data()
        assert os.path.exists("data")
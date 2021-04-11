import unittest
import shutil
from tempfile import mkdtemp
import os
from adatasets import (
    train_test_split,
    fetch_dataset
)


class TestAdatasets(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_path = mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir_path)

    def test_fetch_dataset(self):
        url = 'https://github.com/curran/data/blob/gh-pages/bokeh/IBM.csv'
        # url_target = 'https://raw.githubusercontent.com/curran/data/gh-pages/bokeh/IBM.csv'

        download_path = self.tmp_dir_path
        # harcoding the expected name for this test url
        filedir = 'github_-curran-data-gh-pages-boke'
        filename = 'IBM.csv'
        filepath_exp = os.path.join(download_path, filedir, filename)
        filepath_act = fetch_dataset(
            url_dataset=url, path_target=download_path)
        # checking the return value
        self.assertEqual(filepath_exp, filepath_act)
        file_present = os.path.exists(filepath_exp)
        # checking for actual file existence
        self.assertTrue(file_present)


if __name__ == '__main__':
    unittest.main()

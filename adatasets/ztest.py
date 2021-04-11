# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-

import unittest
import shutil
from tempfile import mkdtemp
import os
from adatasets import (
    test_dataset_regression_fake,
    test_dataset_classification_fake,
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
        # checking for actual file existence
        self.assertTrue(os.path.exists(filepath_exp))

    def test_dataset_classification_fake(self):
        pass


if __name__ == '__main__':
    unittest.main()

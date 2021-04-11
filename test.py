# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import inspect
import random


def log(*s):
    print(*s, flush=True)


##################################################################################################
def test1():
    from adatasets import (
        test_dataset_classification_fake, 
        test_dataset_regression_fake,
        test_dataset_classifier_covtype, 
        test_dataset_classification_petfinder
    )

    ll = ['test_dataset_classification_fake', 'test_dataset_regression_fake',
          'test_dataset_classifier_covtype', 'test_dataset_classification_petfinder'
          ]

    for t in ll:
        log("\n\n##########################", t)
        myfun = locals()[t]
        df, pars = myfun(100)
        log(t, "\n", df, pars)

        assert len(df.index) == 100, 'mismatch test_dataset_classification_fake row data ' + \
            str(len(df.index)) + " vs "+str(nrow)
        assert len(df[pars["colnum"]]) > 0, "error"
        assert len(df[pars["colcat"]]) > 0, "error"


def test2():
    from adatasets import fetch_dataset
    url = 'https://github.com/curran/data/blob/gh-pages/bokeh/IBM.csv'
    download_path = 'ztmp'
    filedir = 'github_curran-data-raw-gh-pages-boke'
    filename = 'IBM.csv'
    filepath_exp = os.path.join(
        os.path.abspath(download_path), filedir, filename)
    filepath_act = fetch_dataset(url_dataset=url, path_target='ztmp')
    assert filepath_act == filepath_exp, f"file path returned by the function '{filepath_act}' is not as expected '{filepath_exp}'"
    assert os.path.exists(filepath_exp)

def test3():
    from adatasets import 

if __name__ == "__main__":
    import fire
    # fire.Fire(test1)
    fire.Fire(test2)

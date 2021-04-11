# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import inspect
import random

##################################################################################################
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
    from adatasets import (
        test_dataset_regression_fake,
        test_dataset_classification_fake
    )
    n_rows = 200
    n_features = 19

    df, pars = test_dataset_regression_fake(n_rows, n_features=n_features)
    assert len(df.index) == n_rows, 'mismatch test_dataset_regression_fake row data ' + str(len(df.index)) + " vs "+str(n_row)
    assert len(pars['colnum']) == n_features, f"numbers of features passed {n_features} is different from return {len(pars['colnum'])}"

    df, pars = test_dataset_classification_fake(n_rows, ndim=n_features)
    assert len(df.index) == n_rows, 'mismatch test_dataset_classification_fake row data ' + str(len(df.index)) + " vs "+str(n_row)
    assert len(pars['colnum']) == n_features, f"numbers of features passed {n_features} is different from return {len(pars['colnum'])}"

def test4():
    from adatasets import test_dataset_classification_petfinder
    n_rows = 500
    df, pars = test_dataset_classification_petfinder(n_rows)
    
    assert 'y' in df.columns, "target column not presnent"


def test5():
    from adatasets import test_dataset_classifier_covtype
    n_rows = 500

    df, pars = test_dataset_classifier_covtype(n_rows)
    
    assert len(df[pars['coly']]) >= 0, "target column should not be empty"


if __name__ == "__main__":
    import fire
    # fire.Fire(test1)
    fire.Fire(test5)

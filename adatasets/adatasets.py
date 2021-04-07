# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os, sys, time, datetime,inspect, json, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

from utilmy.utilmy import (os_makedirs, os_system, global_verbosity, git_current_hash, git_repo_root
                           )

####################################################################################################
verbosity = 3

def log(*s):
    if verbosity>= 1: print(*s, flush=True)

def log2(*s):
    if verbosity>= 1: print(*s, flush=True)


####################################################################################################
def dataset_classifier_XXXXX(nrows=500, **kw):
    """

    """
    colnum = []
    colcat = []
    coly = []
    df = pd.DataFrame
    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly, 'info': '' }
    return df, pars  




####################################################################################################
def split(df, coly=None):
    X,y = df.drop(coly), df[[coly]]
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021)
    return X_train, X_valid, y_train, y_valid, X_test, y_test




####################################################################################################
####################################################################################################
def dataset_classifier_pmlb(name='', return_X_y=False):
    from pmlb import fetch_data, classification_dataset_names
    ds = classification_dataset_names[name]
    pars = {}

    X,y = fetch_data(ds, return_X_y=  True)
    X['coly'] = y
    return X, pars


def test_dataset_classifier_covtype(nrows=500):
    import wget
    # Dense features
    colnum = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",]

    # Sparse features
    colcat = ["Wilderness_Area1",  "Wilderness_Area2", "Wilderness_Area3",
        "Wilderness_Area4",  "Soil_Type1",  "Soil_Type2",  "Soil_Type3",
        "Soil_Type4",  "Soil_Type5",  "Soil_Type6",  "Soil_Type7",  "Soil_Type8",  "Soil_Type9",  ]

    # Target column
    coly        = ["Covertype"]

    log("start")

    root     = os.path.join(os.getcwd() ,"ztmp")
    BASE_DIR = Path.home().joinpath( root, 'data/input/covtype/')
    datafile = BASE_DIR.joinpath('covtype.data.gz')
    datafile.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"

    # Download the dataset in case it's missing
    if not datafile.exists():
        wget.download(url, datafile.as_posix())

    # Read nrows of only the given columns
    feature_columns = colnum + colcat + coly
    df = pd.read_csv(datafile, header=None, names=feature_columns, nrows=nrows)
    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly }

    return df, pars



def test_dataset_regression_fake(nrows=500, n_features=17):
    from sklearn import datasets as sklearn_datasets
    coly   = 'y'
    colnum = ["colnum_" +str(i) for i in range(0, 17) ]
    colcat = ['colcat_1']
    X, y    = sklearn_datasets.make_regression( n_samples=nrows, n_features=n_features, n_targets=1,
                                                n_informative=n_features-1)
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[ci] = np.random.randint(0,1, len(df))

    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly }
    return df, pars


def test_dataset_classification_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    ndim    =11
    coly    = 'y'
    colnum  = ["colnum_" +str(i) for i in range(0, ndim) ]
    colcat  = ['colcat_1']
    X, y    = sklearn_datasets.make_classification(n_samples=1000, n_features=ndim, n_classes=1,
                                                   n_informative=ndim-2)
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[ci] = np.random.randint(0,1, len(df))

    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly }
    return df, pars


def test_dataset_classification_petfinder(nrows=1000):
    import tensorflow as tf
    # Dense features
    colnum = ['PhotoAmt', 'Fee','Age' ]

    # Sparse features
    colcat = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize','FurLength', 'Vaccinated', 'Sterilized',
              'Health', 'Breed1' ]

    colembed = ['Breed1']
    # Target column
    coly        = "y"

    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file    = 'datasets/petfinder-mini/petfinder-mini.csv'
    tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,extract=True, cache_dir='.')

    print('Data Frame Loaded')
    df      = pd.read_csv(csv_file)
    df      = df.iloc[:nrows, :]
    df['y'] = np.where(df['AdoptionSpeed']==4, 0, 1)
    df      = df.drop(columns=['AdoptionSpeed', 'Description'])

    log2(df.dtypes)
    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly, 'colembed' : colembed }
    return df, pars




###################################################################################################
def fetch_dataset(url_dataset, path_target=None, file_target=None):
    """Fetch dataset from a given URL and save it.

    Currently `github`, `gdrive` and `dropbox` are the only supported sources of
    data. Also only zip files are supported.

    :param url_dataset:   URL to send
    :param path_target:   Path to save dataset
    :param file_target:   File to save dataset

    """
    log("###### Download ##################################################")
    from tempfile import mktemp, mkdtemp
    from urllib.parse import urlparse, parse_qs
    import pathlib
    fallback_name        = "features"
    download_path        = path_target
    supported_extensions = [ ".zip" ]

    if path_target is None:
        path_target   = mkdtemp(dir=os.path.curdir)
        download_path = path_target
    else:
        pathlib.Path(path_target).mkdir(parents=True, exist_ok=True)

    if file_target is None:
        file_target = fallback_name # mktemp(dir="")


    if "github.com" in url_dataset:
        """
                # https://github.com/arita37/dsa2_data/raw/main/input/titanic/train/features.zip
              https://github.com/arita37/dsa2_data/raw/main/input/titanic/train/features.zip            
              https://raw.githubusercontent.com/arita37/dsa2_data/main/input/titanic/train/features.csv            
              https://raw.githubusercontent.com/arita37/dsa2_data/tree/main/input/titanic/train/features.zip             
              https://github.com/arita37/dsa2_data/blob/main/input/titanic/train/features.zip
                 
        """
        # urlx = url_dataset.replace(  "github.com", "raw.githubusercontent.com" )
        urlx = url_dataset.replace("/blob/", "/raw/")
        urlx = urlx.replace("/tree/", "/raw/")
        log(urlx)

        urlpath = urlx.replace("https://github.com/", "github_")
        urlpath = urlpath.split("/")
        fname = urlpath[-1]  ## filaneme
        fpath = "-".join(urlpath[:-1])[:-1]   ### prefix path normalized
        assert "." in fname, f"No filename in the url {urlx}"

        os.makedirs(download_path + "/" + fpath, exist_ok= True)
        full_filename = os.path.abspath( download_path + "/" + fpath + "/" + fname )
        log('#### Download saving in ', full_filename)

        import requests
        with requests.Session() as s:
            res = s.get(urlx)
            if res.ok:
                print(res.ok)
                with open(full_filename, "wb") as f:
                    f.write(res.content)
            else:
                raise res.raise_for_status()
        return full_filename



################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()





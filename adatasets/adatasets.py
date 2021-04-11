# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


####################################################################################################
verbosity = 3


def log(*s):
    if verbosity >= 1:
        print(*s, flush=True)


####################################################################################################
# def dataset_classifier_XXXXX(nrows=500, **kw):
#     """
#      Template function for the dataset functions
#     """
#     colnum = []
#     colcat = []
#     coly = []   # target/y column of df
#     df = pd.DataFrame
#     pars = {'colnum': colnum, 'colcat': colcat, "coly": coly, 'info': ''}
#     return df, pars


####################################################################################################
def pd_train_test_split(df, coly=None):
    X, y = df.drop(coly), df[[coly]]
    # making a test train split on all data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.05, random_state=2021)
    # making a validation and train split on previous train data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, random_state=2021)
    return X_train, X_valid, y_train, y_valid, X_test, y_test


####################################################################################################
# def dataset_classifier_pmlb(name='tokyo1', return_X_y=False):
#     """
#     A simple pmlb wrapper to download a specified classification
#     dataset and returns a single dataframe with target
#     column labled as 'y'

#     # what is the point of having this function,
#     # when pmlb is simple on its own
#     """
#     from pmlb import fetch_data, classification_dataset_names
#     ds = classification_dataset_names[name]

#     pars = {}

#     X, y = fetch_data(ds, return_X_y=True)
#     X['coly'] = y
#     return X, pars


def test_dataset_regression_fake(nrows=500, n_features=17):
    """
    working fine
    """
    from sklearn import datasets as sklearn_datasets
    coly = 'y'
    colnum = ["colnum_" + str(i) for i in range(0, n_features)]
    colcat = ['colcat_1']
    X, y = sklearn_datasets.make_regression(n_samples=nrows, n_features=n_features, n_targets=1,
                                            n_informative=n_features-1)
    df = pd.DataFrame(X,  columns=colnum)
    for ci in colcat:
        df[ci] = np.random.randint(0, 1, len(df))
    df[coly] = y.reshape(-1, 1)

    pars = {'colnum': colnum, 'colcat': colcat, "coly": coly}
    return df, pars


def test_dataset_classification_fake(nrows=500, n_features=11):
    """
    working fine
    """
    from sklearn import datasets as sklearn_datasets
    coly = 'y'
    colnum = ["colnum_" + str(i) for i in range(0, n_features)]
    colcat = ['colcat_1']
    X, y = sklearn_datasets.make_classification(n_samples=1000, n_features=n_features, n_classes=1,
                                                n_informative=n_features-2)
    df = pd.DataFrame(X,  columns=colnum)
    for ci in colcat:
        df[ci] = np.random.randint(0, 1, len(df))
    df[coly] = y.reshape(-1, 1)

    pars = {'colnum': colnum, 'colcat': colcat, "coly": coly}
    return df, pars


def test_dataset_classifier_covtype(nrows=500):
    """
    working fine
    """
    log("start")

    import wget
    # Dense features
    colnum = ["Elevation", "Aspect", "Slope",
              "Horizontal_Distance_To_Hydrology", ]

    # Sparse features
    colcat = ["Wilderness_Area1",  "Wilderness_Area2", "Wilderness_Area3",
              "Wilderness_Area4",  "Soil_Type1",  "Soil_Type2",  "Soil_Type3",
              "Soil_Type4",  "Soil_Type5",  "Soil_Type6",  "Soil_Type7",  "Soil_Type8",  "Soil_Type9", ]

    # Target column
    coly = ["Covertype"]

    feature_columns = colnum + colcat + coly

    datafile = os.path.join(os.getcwd() + "/ztmp/covtype/covtype.data.gz")
    datafile_dir = os.path.dirname(datafile)
    if not os.path.exists(datafile_dir):
        os.makedirs(datafile_dir)

    if not Path(datafile).exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        wget.download(url, datafile)

    # Read nrows of only the given columns
    df = pd.read_csv(datafile, header=None, names=feature_columns, nrows=nrows)
    pars = {'colnum': colnum, 'colcat': colcat, "coly": coly}

    return df, pars


def test_dataset_classification_petfinder(nrows=1000):
    """
    working fine
    """
    import zipfile
    import wget

    # Dense features
    colnum = ['PhotoAmt', 'Fee', 'Age']

    # Sparse features
    colcat = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize', 'FurLength', 'Vaccinated', 'Sterilized',
              'Health', 'Breed1']

    colembed = ['Breed1']

    coly = "y"

    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    filepath = os.path.abspath('ztmp/petfinder-mini/')
    filename_csv = "petfinder-mini.csv"
    filename_zip = "petfinder-mini.zip"

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # check if the csv file is there
    if not os.path.exists(os.path.join(filepath, filename_csv)):
        # check if the zip file is present
        if not os.path.exists(os.path.join(filepath, filename_zip)):
            wget.download(dataset_url, os.path.join(
                filepath, filename_zip))
        with zipfile.ZipFile(os.path.join(filepath, filename_zip)) as zip_ref:
            zip_ref.extract('petfinder-mini/petfinder-mini.csv',
                            os.path.join(filepath, '/../'))

    log('Data Frame Loaded')
    df = pd.read_csv(os.path.join(filepath, filename_csv))
    df = df.iloc[:nrows, :]
    df[coly] = np.where(df['AdoptionSpeed'] == 4, 0, 1)
    df = df.drop(columns=['AdoptionSpeed', 'Description'])

    pars = {'colnum': colnum, 'colcat': colcat,
            "coly": coly, 'colembed': colembed}
    return df, pars


###################################################################################################
def fetch_dataset(url_dataset, path_target=None):
    """Fetch dataset from a given URL and save it.

    Currently `github`, `gdrive` and `dropbox` are the only supported sources of
    data. Also only zip files are supported.

    :param url_dataset:   URL to send
    :param path_target:   Path to save dataset
    :param file_target:   File to save dataset

    """
    import requests
    # log("###### Downloading ##################################################")
    supported_extensions = [".csv", ".zip"]

    if path_target is None:
        download_path = os.path.join(os.getcwd(), 'ztmp')
    else:
        download_path = path_target
        os.makedirs(path_target, exist_ok=True)

    # using a filename from the url given, no need for this?
    # fallback_name = "features"
    # if file_target is None:
    #     file_target = fallback_name  # mktemp(dir="")

    if "github.com" in url_dataset:
        """
        Converts github.com domain to raw.githubusercontent.com
        and downloads the file.
        """
        urlx = url_dataset.replace("github.com", "raw.githubusercontent.com")
        urlx = urlx.replace("/blob", "")
        # log(urlx)

        urlpath = urlx.replace("https://raw.githubusercontent.com", "github_")
        urlpath = urlpath.split("/")
        fname = urlpath[-1]  # filename
        fpath = "-".join(urlpath[:-1])[:-1]  # prefix path normalized
        assert "." in fname, f"No filename in the url {urlx}"

        os.makedirs(os.path.join(download_path, fpath), exist_ok=True)
        full_filename = os.path.abspath(
            os.path.join(download_path, fpath, fname))
        # log('#### Download saving in ', full_filename)

        with requests.Session() as s:
            res = s.get(urlx)
            if res.ok:
                print("Successfully downloaded file", res.ok)
                with open(full_filename, "wb") as f:
                    f.write(res.content)
            else:
                raise res.raise_for_status()
        return full_filename


################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()

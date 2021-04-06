# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os, sys, time, datetime,inspect, json, yaml, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split,
from pathlib import Path

####################################################################################################
verbosity = 3

def log(*s):
    if verbosity>= 1: print(*s, flush=True)

def log2(*s):
    if verbosity>= 1: print(*s, flush=True)

def log3(*s):
    if verbosity>= 1: print(*s, flush=True)


####################################################################################################


def dataset_classifier_XXXXX(nrows=500, **kw):

    ....

    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly, 'info': '' }
    return df, pars  










####################################################################################################
def split(df):
    X,y = df
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021)



def dataset_classifier_pmlb(name='', return_X_y=False):
    from pmlb import fetch_data, classification_dataset_names

    ds = classification_dataset_names[name]:

    df = fetch_data(ds, return_X_y= return_X_y)
    return df, pars



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



def test_dataset_regress_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    coly   = 'y'
    colnum = ["colnum_" +str(i) for i in range(0, 17) ]
    colcat = ['colcat_1']
    X, y    = sklearn_datasets.make_regression( n_samples=1000, n_features=17, n_targets=1, n_informative=17)
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[colcat] = np.random.randint(0,1, len(df))

    return df, colnum, colcat, coly




def test_dataset_classi_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    ndim    =11
    coly    = 'y'
    colnum  = ["colnum_" +str(i) for i in range(0, ndim) ]
    colcat  = ['colcat_1']
    X, y    = sklearn_datasets.make_classification(n_samples=1000, n_features=ndim, n_targets=1, n_informative=ndim
    )
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[colcat] = np.random.randint(0,1, len(df))

    return df, colnum, colcat, coly


def test_dataset_petfinder(nrows=1000):
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

    print(df.dtypes)
    return df, colnum, colcat, coly, colembed




###################################################################################################
def git_repo_root():
    try :
      cmd = "git rev-parse --show-toplevel"
      mout, merr = os_system(cmd)
      path = mout.split("\n")[0]
      if len(path) < 1:  return None
    except : return None    
    return path

    
def git_current_hash(mode='full'):
   import subprocess 
   # label = subprocess.check_output(["git", "describe", "--always"]).strip();   
   label = subprocess.check_output([ 'git', 'rev-parse', 'HEAD' ]).strip();      
   label = label.decode('utf-8')
   return label








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
def os_get_function_name():
    return sys._getframe(1).f_code.co_name


def os_getcwd():
    root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
    return  root


def os_system(cmd, doprint=False):
  """ get values
       os_system( f"   ztmp ",  doprint=True)
  """
  import subprocess  
  try :
    p          = subprocess.run( cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, )
    mout, merr = p.stdout.decode('utf-8'), p.stderr.decode('utf-8')    
    if doprint: 
      l = mout  if len(merr) < 1 else mout + "\n\nbash_error:\n" + merr
      print(l)

    return mout, merr
  except Exception as e :
    print( f"Error {cmd}, {e}")

      
def os_makedirs(dir_or_file):
    if os.path.isfile(dir_or_file) :
        os.makedirs(os.path.dirname(os.path.abspath(dir_or_file)), exist_ok=True)
    else :
        os.makedirs(os.path.abspath(dir_or_file), exist_ok=True)









################################################################################################
class dict_to_namespace(object):
    def __init__(self, d):
        self.__dict__ = d











##################################################################################################
def test():
   from utilmy import (os_makedirs, Session, global_verbosity, os_system  
                       
                      )

  
if __name__ == "__main__":
    import fire
    fire.Fire()





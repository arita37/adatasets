# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os, sys, time, datetime,inspect

def log(*s):
    print(*s, flush=True)



##################################################################################################
def test1():
   from adatasets import (test_dataset_classification_fake, test_dataset_regression_fake,
    test_dataset_classifier_covtype, test_dataset_classification_petfinder

   )

   ll = [ 'test_dataset_classification_fake', 'test_dataset_regression_fake',
          'test_dataset_classifier_covtype', 'test_dataset_classification_petfinder'
   ]

   for t in ll :
     log("\n\n##########################", t)
     myfun    = locals()[t]
     df, pars = myfun(100)
     log(t, "\n", df, pars)



if __name__ == "__main__":
    import fire
    fire.Fire(test1)






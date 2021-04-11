# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os, sys, time, datetime,inspect,random

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
   nrows= 100
   for t in ll :
     log("\n\n##########################", t)
     myfun    = locals()[t]
     df, pars = myfun(nrows)
     log(t, "\n", df, pars)
     
     assert len(df.index) == 100, f'mismatch {t} row data {len(df)}  vs {nrows}'


     log("\n######### Column names check ")
     cols_family = {}
     for key,val in pars.items():
         if key.startswith('col') :
             cols_family[key] =  val

     for colname, colist in cols_family.items():
        print(colname, colist)
        assert len(df[ colist ]) > 0 , f"missing {colname}"







if __name__ == "__main__":
    import fire
    fire.Fire(test1)






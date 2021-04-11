import random
from unittest import TestCase
from adatasets import test_dataset_classification_fake, test_dataset_regression_fake,test_dataset_classifier_covtype, test_dataset_classification_petfinder

import adatasets as ds

#ds.pd_train_test_split()
#ds.pd_train_test_split2()
#df, pr =ds.dataset_classifier_pmlb()


#df, pr =ds.test_dataset_classifier_covtype()
# df1, pr1 =ds.test_dataset_regression_fake()
# df2, pr2 =ds.test_dataset_classification_fake()
# df3, pr3 =ds.test_dataset_classification_petfinder()

def log(*s):
    print(*s, flush=True)



class test_dataset_class(TestCase):
    pass

    def test_test_dataset_classification_fake(self):
        nrow = random.randint(0,20)*100
        df, pars = test_dataset_classification_fake(nrow)
        assert len(df.index) == nrow, 'mismatch test_dataset_classification_fake row data '+str(len(df.index)) +" vs "+str(nrow)

    def test_test_dataset_regression_fake(self):
        nrow = random.randint(0,20)*100
        df, pars =  test_dataset_regression_fake(nrow)
        assert len(df.index) == nrow, 'mismatch test_dataset_regression_fake row data'+str(len(df.index)) +"vs"+str(nrow)

    def test_test_dataset_classifier_covtype(self):
        nrow = random.randint(0,20)*100
        df, pars = test_dataset_classifier_covtype(nrow)
        assert len(df.index) == nrow, 'mismatch test_dataset_classifier_covtype row data'+str(len(df.index)) +"vs"+str(nrow)

    def test_test_dataset_classification_petfinder(self):
        nrow = random.randint(0,20)*100
        df, pars = test_dataset_classification_petfinder(nrow)
        assert len(df.index) == nrow, 'mismatch test_dataset_classification_petfinder row data'+str(len(df.index)) +"vs"+str(nrow)


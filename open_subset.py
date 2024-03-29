import numpy as np
import pandas as pd
import parsers as pa
from dateutil.relativedelta import relativedelta

import pdb

'''
INSTRUCTIONS FOR USING COMPLETE OPEN_AND_PROCESS SYSTEM:
for creating training set:
    type = "transform", subset_type = "model", model_purpose = "train"
for post training, first testing:
    type = "transform", subset_type = "model", model_purpose = "test1"
for dynamic testing:
    type = "transform", subset_type = "model", model_purpose = "test2"
    ##>> may need to add one transformation to take care of "current" vs "fully paid"
for analysis
    type = "transform", subset_type = "analysis"
'''


# create file that opens file, returning specified type
def open_and_process(file, type="simple", subset_type="model", model_purpose=None):
    print "OPEN beginning"
    # basic open into pandas
    data = pd.read_csv(open(file))

    # options are to help with debugging, can re-select simpler subsetting/transformations as needed
    if type == "simple":
        return data
    elif type == "feature_subset":
        return subset_by_feature(data, subset_type)
    elif type == "transform":
        #pdb.set_trace()
        data = subset_by_feature(data, subset_type)
        data = transform(data, subset_type)
        return data
    elif type == "tree_transform":
        #pdb.set_trace()
        data = subset_by_feature(data, subset_type)
        data = tree_transform(data, subset_type)
        return data
    else:
        return "ERROR: unknown type used"


def subset_by_feature(db, subset_type="model"):
    print "SUBSET beginning"
    print "current db shape: " + str(db.shape)
    # take only features of interest
    
    if subset_type is "model":
        # take only features of interest for modeling (training or testing)
        return db.loc[:,['title','desc','purpose','loan_amnt','term','int_rate','home_ownership','fico_range_high','fico_range_low','dti','is_inc_v','annual_inc','emp_length','issue_d','loan_status']]
    elif subset_type is "model2":
        # take only features of interest for modeling (training or testing)
        return db.loc[:,['title','loan_amnt','int_rate','home_ownership','fico_range_low','dti','is_inc_v','annual_inc','emp_length','issue_d','loan_status']]
        # implement 'model2' -> match other feature sets recommended
        # reduce complexity where possible in the features
        # run a pre-test on features through sklearn
    elif subset_type is "tree_model":
        # take only features of interest for modeling (training or testing)
        return db.loc[:,['title','desc','purpose','loan_amnt','term','int_rate','home_ownership','fico_range_low','fico_range_high','dti','is_inc_v','annual_inc','emp_length','issue_d','loan_status']]
    elif subset_type is "tree_model_2":
        # take only features of interest for modeling (training or testing)
        return db.loc[:,['title','desc','purpose','loan_amnt','term','int_rate','home_ownership','fico_range_low','fico_range_high','dti','is_inc_v','annual_inc','emp_length','issue_d','loan_status']]
    elif subset_type is "analysis":
        # take only features of interest for analyzing results of historical data vs choices by the model
        return db.loc[:,['grade','loan_amnt','int_rate','total_rec_int','total_rec_prncp','installment','issue_d','last_pymnt_d','loan_status','term']]
    else:
        return "subset_type " + subset_type + " does not exist.  Use either 'model_creation' or 'default_analysis'."

# 'acc_open_past_24mths' has a lot of empty ones early in the dataset

def transform(db, subset_type="model"):
    print "TRANSFORM beginning"
    print "current db shape: " + str(db.shape)

    ##### transformations to complete before droping NA #####
    if 'title' in db.columns:
        # change title to title_length (NaN to 0)
        db['title'] = db['title'].fillna("").apply(len)
        # remove all titles over len(40); seems like not allowed
        db = db[db['title'] <= 40]
    if 'desc' in db.columns:
        # change desc to desc_length (NaN to 0)
        db['desc'] = db['desc'].fillna("").apply(len)
    if 'purpose' in db.columns:
        # for purpose, NaN to "not_given"
        db['purpose'] = db['purpose'].fillna('not_given')
    print "before dropna, db length: " + str(len(db))
    # remove remaining NaN
    db = db.dropna(axis=0)
    print "just after dropna, db length: " + str(len(db))
    ##### transformations to complete after droping NA #####
    if 'term' in db.columns:
        # transform 'term' into continuous numeric values
        db['term'] = db['term'].apply(pa.to_num)
    if 'int_rate' in db.columns:
        # transform 'int_rate' into continuous numeric values
        db['int_rate'] = db['int_rate'].apply(pa.to_num)
    if 'annual_inc' in db.columns:
        # remove all with income over 150000 (outliers)
        db = db[db['annual_inc'] <= 150000]
    if 'emp_length' in db.columns:
        # make number of years working into continuous
        db['emp_length'] = db['emp_length'].apply(pa.parse_employment_years)
    if 'is_inc_v' in db.columns:
        # change 'is_inc_v' to numerical version of T/F
        db['is_inc_v'] = db['is_inc_v'].apply(pa.binary_true_false)
    if 'exp_d' in db.columns:
        db['exp_d'] = db['exp_d'].apply(pa.parse_date)
    if 'issue_d' in db.columns:
        db['issue_d'] = db['issue_d'].apply(pa.parse_date)
    if 'last_pymnt_d' in db.columns:
        db['last_pymnt_d'] = db['last_pymnt_d'].apply(pa.parse_date)
    # change categorical variables into dummy varialbes
    if 'home_ownership' in db.columns:
        db = pa.categorical_transform(db, 'home_ownership', 'home_own')
        #del db['home_own_RENT']
        #del db['home_own_NONE']
        #del db['home_own_OTHER']
    if 'purpose' in db.columns:
        db = pa.categorical_transform(db, 'purpose')
    # only transform categories if asked to
    if (subset_type is "model") or (subset_type is "model2"):
        print "grade and loan_status transformations occuring"
        # only need to do categorical transformatings for modeling creation and prediction purposes
        if 'grade' in db.columns:
            db = pa.categorical_transform(db, 'grade')
        if 'loan_status' in db.columns:
            # transoform loan_status (variable to predict)
            db = pa.LC_status_transform_new(db)
    elif subset_type is "analysis":
        db = group_prep(db)
        #db = calc_loan_age(db)
    return db


def tree_transform(db, subset_type="model"):
    print "TRANSFORM beginning"
    print "current db shape: " + str(db.shape)

    ##### transformations to complete before droping NA #####
    if 'title' in db.columns:
        # change title to title_length (NaN to 0)
        db['title'] = db['title'].fillna("").apply(len)
        # remove all titles over len(40); seems like not allowed
        db = db[db['title'] <= 40]
    if 'desc' in db.columns:
        # change desc to desc_length (NaN to 0)
        db['desc'] = db['desc'].fillna("").apply(len)
    if 'purpose' in db.columns:
        # for purpose, NaN to "not_given"
        db['purpose'] = db['purpose'].fillna('not_given')
    print "before dropna, db length: " + str(len(db))
    # remove remaining NaN
    db = db.dropna(axis=0)
    print "just after dropna, db length: " + str(len(db))
    ##### transformations to complete after droping NA #####
    if 'term' in db.columns:
        # transform 'term' into continuous numeric values
        db['term'] = db['term'].apply(pa.to_num)
    if 'int_rate' in db.columns:
        # transform 'int_rate' into continuous numeric values
        db['int_rate'] = db['int_rate'].apply(pa.to_num)
    if 'annual_inc' in db.columns:
        # remove all with income over 150000 (outliers)
        db = db[db['annual_inc'] <= 150000]
    if 'emp_length' in db.columns:
        # make number of years working into continuous
        db['emp_length'] = db['emp_length'].apply(pa.parse_employment_years)
    if 'is_inc_v' in db.columns:
        # change 'is_inc_v' to numerical version of T/F
        db['is_inc_v'] = db['is_inc_v'].apply(pa.binary_true_false)
    if 'exp_d' in db.columns:
        db['exp_d'] = db['exp_d'].apply(pa.parse_date)
    if 'issue_d' in db.columns:
        db['issue_d'] = db['issue_d'].apply(pa.parse_date)
    if 'last_pymnt_d' in db.columns:
        db['last_pymnt_d'] = db['last_pymnt_d'].apply(pa.parse_date)
    # change categorical variables into dummy varialbes
    if 'home_ownership' in db.columns:
        db = pa.categorical_transform(db, 'home_ownership', 'home_own')
        #del db['home_own_RENT']
        #del db['home_own_NONE']
        #del db['home_own_OTHER']
    if 'purpose' in db.columns:
        db = pa.categorical_transform(db, 'purpose')
    # only transform categories if asked to
    #if (subset_type is "model") or (subset_type is "model2"):
    #    print "grade and loan_status transformations occuring"
        # only need to do categorical transformatings for modeling creation and prediction purposes
    #    if 'grade' in db.columns:
    #        db = pa.categorical_transform(db, 'grade')
    #    if 'loan_status' in db.columns:
            # transoform loan_status (variable to predict)
    #        db = pa.LC_status_transform_new(db)
    db = pa.LC_status_transform_new(db)

    if subset_type is "analysis":
        db = group_prep(db)
        #db = calc_loan_age(db)
    return db


def group_prep(db):
    db['yy-mm_start_date'] = db['issue_d'].apply(pa.yr_mm_grouping)
    db['yy-mm_last_payment'] = db['last_pymnt_d'].apply(pa.yr_mm_grouping)
    return db

def calc_loan_age(db):
    ###  NOTE - must first transform date strings into Python datetime objects
    # create new 'empty' columns putting results of calculations in
    db['final_age'] = 0
    db['last_pay_date_and_age'] = "0"
    # initialize count to keep track of progress
    count = 0
    # run analysis (using iterrows because relativedelta() has error when running )
    print "running analysis"
    for row, values in db.iterrows():
        age = relativedelta(db.loc[row,'last_pymnt_d'],db.loc[row,'issue_d'])
        months = age.years*12 + age.months
        db.loc[row, 'final_age'] = months
        db.loc[row, 'last_pay_date_and_age'] = db.loc[row, 'yy-mm_last_payment'] + '_' + str(months)
        count += 1
        if count % 2000 == 0:
            print count
    return db
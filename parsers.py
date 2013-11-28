from datetime import datetime
import re
import numpy as np
import pandas as pd

import pdb


def to_num(x):
    try:
        return float(re.sub(r'[^0-9\.]', '', x))
    except ValueError:
        return "Error: do not proceed"


def binary_true_false(x):
    if x != False:
        return 1
    return 0
    # note, can use '.fillna(False)' before this to have all NaN be 0(False), and everything else be 1(True)


def parse_employment_years(x):
    sp = re.split(" ", x)
    if sp:
        if sp[0] == '<':
            return float(0.5)
        elif sp[0] == '10+':
            return float(10)
        elif sp[0] == 'n/a':
            return float(0)
        return float(sp[0])
    return None

def categorical_transform(data_b, column_name, pref=None):
    # column = column to transform
    # make copy to not impact original database
    db = data_b.copy()
    # if not given, define the prefix as the original column name
    if pref == None:
        pref = column_name
    # do get_dummies transform
    new = pd.get_dummies(db[column_name], prefix=pref)
    # append new columns onto db
    for each in new.columns:
        db[each] = new[each]
    # delete original column
    del db[column_name]
    return db


def parse_date(x, format='%Y-%m-%d'):
    d = datetime.strptime(x, format)
    return d

def yr_mm_grouping(x):
    # NOTE: x must be a timestamp object
    month = str(x.month)
    # add 0 to the beginning of single digit months
    if len(month) is 1:
        month = "0" + month
    return str(x.year) + "-" + month


def LC_status_transform(data_b):
    db = data_b.copy()
    ls = "loan_status"
    print "database length: " + str(len(db))
    # group Default and Charged Off together
    db[ls] = db[ls].apply(replace_chTOde)
    db[ls] = db[ls].apply(replace_curTOfull)
    # get rid of unwanted variables
    db = db[db[ls].isin(['Fully Paid', 'Default'])]
    print "database length after LC_status_transform: " + str(len(db))
    return db

def LC_status_transform_new(data_b):
    db = data_b.copy()
    ls = "loan_status"
    print "database length: " + str(len(db))
    # group Default and Charged Off together
    db[ls] = db[ls].apply(replace_to_poor_standing)
    db[ls] = db[ls].apply(replace_to_good_standing)
    # get rid of unwanted variables
    db = db[db[ls].isin(['in_good_standing', 'in_poor_standing'])]
    db = categorical_transform(db, 'loan_status', 'status')
    del db['status_in_poor_standing']
    print "database length after LC_status_transform: " + str(len(db))
    return db


def replace_to_poor_standing(x):
    poor_standing_list = [
        "Charged Off",
        "Default",
        "Late (16-30 days)",
        "Late (31-120 days)",
        "Does not meet the credit policy.  Status:Charged Off",
        "Does not meet the credit policy.  Status:Late (16-30 days)",
        "Does not meet the credit policy.  Status:Late (31-120 days)"
        ]
    if x in poor_standing_list:
        return "in_poor_standing"
    return x


def replace_to_good_standing(x):
    good_standing_list = [
        "Current",
        "Fully Paid",
        "In Grace Period",
        "Does not meet the credit policy.  Status:Current",
        "Does not meet the credit policy.  Status:Fully Paid",
        "Does not meet the credit policy.  Status:In Grace Period"
        ]
    if x in good_standing_list:
        return "in_good_standing"
    return x


def replace_mortTOown(x):
    if x == "MORTGAGE":
        return "OWN"
    return x


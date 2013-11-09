from datetime import datetime
import re
import numpy as np
import pandas as pd


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
    # group Default and Charged Off together
    db[ls]= db[ls].apply(replace_chTOde)
    # get rid of unwanted variables
    ##b_1 = db[db['loan_status'] == 'Fully Paid']
    ##db_2 = db[db['loan_status'] == 'Default']
    ##db = pd.concat([db_1, db_2])
    db = db[db[ls].isin(['Fully Paid', 'Default'])]
    return db


def replace_chTOde(x):
    if x == "Charged Off":
        return "Default"
    return x
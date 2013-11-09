import numpy as np
import pandas as pd
import parsers as pa

# create file that opens file, returning specified type
def open_and_process(file, type="simple", source="LC"):
	
	# basic open into pandas
	data = pd.read_csv(open(file))

	if source != "LC":
		return "prosper and other datasets are not yet integrated"

	# options are to help with debugging, can re-select simpler subsetting/transformations as needed
	if type == "simple":
		return data
	elif type == "feature_subset":
		return subset_by_feature(data, source)
	elif type == "transform":
		data = subset_by_feature(data, source)
		return transform(data)
	else:
		return "ERROR: unknown type used"


def subset_by_feature(db, source="LC"):
	# take only features of interest
	
	if source == "LC":
		return db.loc[:,['title','desc','is_inc_v','loan_amnt','term','int_rate','grade','home_ownership','fico_range_high','fico_range_low','dti','purpose','annual_inc','emp_length','loan_status']]
	else:
		return "Prosper subsetting not yet complete"
		# TO DO - how do a send up an excemption instead?


def transform(db, source="LC"):

	if source == "LC":
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
		# remove remaining NaN
		db = db.dropna(axis=0)
		##### transformations to complete after droping NA #####
		if 'term' in db.columns:
			# transform 'term' into continuous numeric values
			db['term'] = db['term'].apply(pa.to_num)
		if 'int_rate' in db.columns:
			# transform 'int_rate' into continuous numeric values
			db['int_rate'] = db['int_rate'].apply(pa.to_num)
		if 'annual_inc' in db.columns:
			# remove all with income over 200000 (outliers)
			db = db[db['annual_inc'] <= 150000]
		if 'emp_length' in db.columns:
			# make number of years working into continuous
			db['emp_length'] = db['emp_length'].apply(pa.parse_employment_years)
		if 'is_inc_v' in db.columns:
			# change 'is_inc_v' to numerical version of T/F
			db['is_inc_v'] = db['is_inc_v'].apply(pa.binary_true_false)
		if 'exp_d' in db.columns:
			db['exp_d'] = db['exp_d'].apply(pa.parse_date)
		if 'last_pymnt_d' in db.columns:
			db['last_pymnt_d'] = db['last_pymnt_d'].apply(pa.parse_date)
		# change categorical variables into dummy varialbes
		if 'home_ownership' in db.columns:
			db = pa.categorical_transform(db, 'home_ownership', 'home_own')
		if 'purpose' in db.columns:
			db = pa.categorical_transform(db, 'purpose')
		#if 'grade' in db.columns:
		#	db = pa.categorical_transform(db, 'grade')
		#if 'loan_status' in db.columns:
		#	# transoform loan_status (variable to predict)
		#	db = pa.LC_status_transform(db)
		#	db = pa.categorical_transform(db, 'loan_status', 'status')
	else:
		return "Prosper transformations not yet complete"
		# TO DO - how do I send up an excemption instead?
	return db


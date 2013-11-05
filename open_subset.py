import numpy as np
import pandas as pd
from parsers import to_num, binary_true_false

# create file that opens and subsets file, returning only entirety of file (but only opens part at a time to reduce RAM usage)
def open_and_subset(file, type="simple", source="LC"):
	
	data = pd.read_csv(open(file))

	if type == "simple":
		return data
	elif type == "feature_only":
		return subset_by_feature(data, source)
	elif type == "through_transform"
		data = subset_by_features(data, source)
		return transform(data)
	else:
		return "ERROR: unknown type used"


def subset_by_feature(db, source="LC"):
	# take only features of interest
	
	if source == "LC":
		return db.loc[:,['title','desc','is_inc_v','loan_amnt','term','int_rate','grade','home_ownership','purpose','annual_inc','loan_status']]
	else:
		return "Prosper subsetting not yet complete"


def transform(db, source="LC"):

	if source == "LC":
		##### complete transformation
		# change title to title_length (NaN to 0)
		db['title'] = db['title'].fillna("").apply(len)
		# change desc to desc_length (NaN to 0)
		db['desc'] = db['desc'].fillna("").apply(len)
		# for purpose, NaN to "not_given"
		db['purpose'] = db['purpose'].fillna('not_given')
		# remove remaining NaN
		db = db.dropna(axis=0)
		##### complete parsing 
		# transform 'term' into continuous numeric values
		db['term'] = db['term'].apply(to_num)
		# transform 'int_rate' into continuous numeric values
		db['int_rate'] = db['int_rate'].apply(to_num)
		# transform 'emp_title' into whether or not employer's title is listed


		return db.dropna(axis=0)



'''
	if source == "LC":
		return db.loc[:,['title','desc','is_inc_v','loan_amnt','term','int_rate','grade','home_ownership','purpose','emp_title','annual_inc','loan_status']]
	elif source == "P":
		return "TBD"
	else:
		return "error"
	#new = new[np.isfinite(new['FICO_NXT_GEN_V2_SCORE_VALUE'])]
'''

########################## NOTES ############################
'''
NOT USING these methods right now
'''

# takes pandas database and spits out subsetted for analysis
def subset_basic(db):
	# only using 'Booked Loans' (not 'Application No Listing', nor 'listing no loan')
	new = db[db['DeclineGroup'] == "Booked Loan"]
	# only use loans Prosper would currently accept (as noted in "Eligible_Oct2013v1")
		# BECAUSE purpose is to increase performance for current investors
	new = new[new["Eligible_Oct2013v1"] == 1]
	return new


'''
LC values removing NaN from:
	> 'is_inc_v' (3 out of 42538)
'''


'''
MISC functions likely to be incorporated
df.replace(1.5, nan, inplace=True)

def subset_full_parse(db):
	# for continuous, simple : change strings to numeric
	# for categorical, simple: change strings to features (0,1 for each )
	# for for categorical to continuous: make specific function for each instance (ex: for_income)


Categories NaN replacements made:
	> 'EmploymentStatus' NaN to 'not_listed'
	> 'Occupation' NaN to 'not_listed'

Categories I'm OK loosing NaN values from for the MVP
	> 'AnnualIncomeRaw'
	> FICO_NXT_GEN_V2_SCORE_VALUE
	> 'ListingTermMonths'
	> 'ListingCategoryName'
	> 'ListingDescriptionLen'

Categories I'm OK loosing NaN values from in final product
	> 'LoanAmount' 

Categories with no apparent NaNs (only subset explicitly checked):
	> 'CurrentDelinquencies'
	> 'DelinquenciesLast7Years'
	> 'IncomeVerifiable'
	> 'TotalCreditLines'

Things to add into subset_advanced
	- aggrigate the scores ahead of time (each feature = a unique vector for each scoring parameter)

'''
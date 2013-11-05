import numpy as np
import pandas as pd

# create file that opens and subsets file, returning only entirety of file (but only opens part at a time to reduce RAM usage)
def open_and_subset(file, type="simple"):
	data = pd.read_csv(open(file))
	if type == "simple":
		return subset_basic(data)
	elif type == "advanced":
		return subset_advanced(data)
	else:
		return "ERROR: no type defined"

# takes pandas database and spits out subsetted for analysis
def subset_basic(db):
	# only using 'Booked Loans' (not 'Application No Listing', nor 'listing no loan')
	new = db[db['DeclineGroup'] == "Booked Loan"]
	# only use loans Prosper would currently accept (as noted in "Eligible_Oct2013v1")
		# BECAUSE purpose is to increase performance for current investors
	new = new[new["Eligible_Oct2013v1"] == 1]
	return new


def subset_advanced(db):
	# do initial basic subset, removing all non-booked loans
	new = subset_basic(db)
	# take only features of interest
	new = new.loc[:,['EmploymentStatus','Occupation','FICO_NXT_GEN_V2_SCORE_VALUE','ListingTermMonths','ListingCategoryName','ListingDescriptionLen','LoanAmount','CurrentDelinquencies','DelinquenciesLast7Years','FirstRecordedCreditLine','IncomeVerifiable','AnnualIncomeRaw','TotalCreditLines']]
	# remove "Exception" from FICO_NEXT_GEN scores
	new = new[new['FICO_NXT_GEN_V2_SCORE_VALUE'] != 'Exception']
	#new = new[np.isfinite(new['FICO_NXT_GEN_V2_SCORE_VALUE'])]
	return new


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
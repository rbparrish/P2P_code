import numpy
import pandas as pd

import parsers as pa

# determines what each column type is.
# ALSO, makes sure there is only one type per column
def check_types(db):
	# make list of all the types in the 
	ans_set = []
	for column in db:
    	 ans_set.append((column, set(db[column].apply(type))))

  	return ans_set


def check_date_range(db):
	if 'yy-mm_start_date' not in db.columns:
		db['yy-mm_start_date'] = db['issue_d'].apply(pa.yr_mm_grouping)
	return set(db['yy-mm_start_date'])



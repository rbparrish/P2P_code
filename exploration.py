import numpy
import pandas as pd

# determines what each column type is.
# ALSO, makes sure there is only one type per column
def check_types(db):
	# make list of all the types in the 
	ans_set = []
	for column in db:
    	 ans_set.append((column, set(db[column].apply(type))))

  	return ans_set


'''
####################### SOME DATA VISUALIZATION EXAMPLES ###############

####  histogram of FICO score
import matplotlib.pyplot as plt
import pandas as pd
plt.figure()
loansmin = pd.read_csv('../datasets/loanf.csv')
fico = loansmin['FICO.Score']
p = fico.hist()



####  Box Plot of FICO score ####
import matplotlib.pyplot as plt
import pandas as pd
plt.figure()
loansmin = pd.read_csv('../datasets/loanf.csv')

p = loansmin.boxplot('Interest.Rate','FICO.Score')
q = p.set_xticklabels(['640','','','','660','','','','680','','','','700',
  '720','','','','740','','','','760','','','','780','','','','800','','','','820','','','','840'])

q0 = p.set_xlabel('FICO Score')
q1 = p.set_ylabel('Interest Rate %')
q2 = p.set_title('                          ')

##### scatterbox matrix of different variables #####
import pandas as pd
loansmin = pd.read_csv('../datasets/loanf.csv')
a = pd.scatter_matrix(loansmin,alpha=0.05,figsize=(10,10), diagonal='hist')
'''
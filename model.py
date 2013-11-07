import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import open_subset as op
import parsers as pa


class Classifier:

	def __init__(self):
		
		#self.file_loc = file_loc
		self.data = None
		self.data_scaled = None
		self.pred = None

		print "ready to open file. Run .open_file().  Type options: simple, feature_subset, transform"

	# open 
	def open_file(self):
		d = op.open_and_process('../Raw_Data/LendingClub/LoanStats3a.csv', 'transform')
		del d['status_Default']
		self.data = d
		print "run .prepare()"

	def prepare(self):
		print "normalizing and separating into training and testing groups"
		# separate out predictive values
		self.pred = self.data['status_Fully Paid']
		del self.data['status_Fully Paid']
		# scale feature variables
		from sklearn import preprocessing
		self. data_scaled = preprocessing.scale(self.data)
		# change predict values into numpy array to work with scikitlearn
		self.pred = self.pred.values
		print "run .log_regression()"

	def assign_to_groups(self):
		##### assign data into bulk training and test groups #####
		# split at sample # 15,000
		n = 15000
		training_features = self.data_scaled[:n]
		training_pred = self.pred[:n]
		test_features = self.data_scaled[n:]
		test_pred = self.pred[n:]
		return training_features, training_pred, test_features, test_pred

	def log_regression(self):
		##### run logistic regression & output result #####
		from sklearn import linear_model
		clf = linear_model.SGDClassifier()
		# get groups & train algorithm
		tr_feat, tr_pred, tst_feat, tst_pred = self.assign_to_groups()
		probas_ = clf.fit(tr_feat, tr_pred)
		# make predictions based on built model
		print "Accuracy of model: ", clf.score(tst_feat, tst_pred)
		p = clf.predict(tst_feat)
		self.display_ROC(p, tst_pred)

	def display_ROC(self, pr, ac):
		##### make ROC graph for the result #####
		from sklearn.metrics import roc_curve, auc
		fpr, tpr, thresholds = roc_curve(pr, ac)
		roc_auc = auc(fpr, tpr)
		# build & display ROC plot (MUST HAVE '%matplotlib inline' if running in iPython)
		plt.clf()
		plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic example')
		plt.legend(loc="lower right")
		plt.show()


'''
	def separate_by_grade(self, tst_feat, tst_pred):
		##### separate test group by grade #####
		grades = ['A','B','C','D','E','F','G']
		for each in grades:

		self.make_prediction(feat, pred)
'''

'''

# We create a function in code that encapsulates all this.
# It takes as input, a borrowers FICO score, the desired loan amount and 
# the coefficient vector from our model. It returns a probability of getting the loan, 
# a number between 0 and 1.
def pz(fico,amt,coeff):
  # compute the linear expression by multipyling the inputs by their respective coefficients.
  # note that the coefficient array has the intercept coefficient at the end
  z = coeff[0]*fico + coeff[1]*amt + coeff[2]
  return 1/(1+exp(-1*z))

'''
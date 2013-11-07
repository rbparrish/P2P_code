import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing

import pdb

import open_subset as op
import parsers as pa


class Classifier:

	def __init__(self):
		
		#self.file_loc = file_loc
		self.data = None
		self.pred = None

		print "ready to open file. Run .open_file().  Type options: simple, feature_subset, transform"

	# open 
	def open_file(self):
		d = op.open_and_process('../Raw_Data/LendingClub/LoanStats3a.csv', 'transform')
		del d['status_Default']
		self.data = d
		print "run .assign_to_groups()"

	def assign_to_groups(self):
		# separate out predictor
		self.pred = self.data['status_Fully Paid']

		# assign each to bulk groups
		n = 15000
		self.tr_feat = self.data[:n]
		del self.tr_feat['status_Fully Paid']
		self.tr_pred = self.pred[:n]
		self.tst_dt = self.data[n:]

	def prepare(self):
		self.scaler = preprocessing.StandardScaler()
		self.tr_feat = self.scaler.fit_transform(self.tr_feat)
		self.tr_pred = self.tr_pred.values

	def run(self, split=None):
		# train logistic regression model
		self.clf = linear_model.SGDClassifier()
		probas_ = self.clf.fit(self.tr_feat, self.tr_pred)
		# do analysis
		if split is "by_grades":
			grades = ['A','B','C','D','E','F','G']
			for letter in grades:
				dt = self.tst_dt[self.tst_dt['grade_'+letter] == 1]
				tst_pred = dt['status_Fully Paid']
				tst_feat = dt
				del tst_feat['status_Fully Paid']
				print "Model performance on notes of grade " + letter
				self.single_analysis((self.scaler.transform(tst_feat)), tst_pred)
		else:
			tst_pred = self.tst_dt['status_Fully Paid']
			tst_feat = self.tst_dt
			del tst_feat['status_Fully Paid']
			self.single_analysis(self.scaler.transform(tst_feat), tst_pred)

	def single_analysis(self, features, true_results):
		# general accuracy of model
		print "Accuracy of model: ", self.clf.score(features, true_results)
		# information on total loan set, by grade
		total_paid = sum(true_results)
		total_notes = len(true_results)
		print "Total Fully Paid in Grade: " + str(total_paid)
		print "Total Notes in Grade: " + str(total_notes)
		print "Rate if fully invest: " + str(float(total_paid)/total_notes)
		# information on performance of the model
		pr = self.clf.predict(features)
		total_investments_by_prediction = sum(pr)
		print "Total invested in by model: " + str(total_investments_by_prediction)
		################
		ac = true_results.values
		correct = 0
		for i in range(len(pr)):
			if (pr[i] == 1) and (ac[i] == 1):
				correct += 1
		print "precent not-defaulted by model: " + str(float(correct)/sum(pr))
		self.display_ROC(pr, ac)



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
	def log_regression(self):
		##### run logistic regression & output result #####


		from sklearn import linear_model
		clf = linear_model.SGDClassifier()
		# get groups & train algorithm
		tr_feat, tr_pred, tst_feat, tst_pred = self.assign_to_groups()
		probas_ = clf.fit(tr_feat, tr_pred)
		# make predictions based on built model
		p = clf.predict(tst_feat)
		self.display_ROC(p, tst_pred)


	def prepare(self):
		#####
		self.assign_to_groups()


		# separate out predictive values
		self.pred = self.data['status_Fully Paid']
		del self.data['status_Fully Paid']
		# scale feature variables
		from sklearn import preprocessing
		self.data_scaled = preprocessing.scale(self.data)
		# change predict values into numpy array to work with scikitlearn
		self.pred = self.pred.values
		print "run .log_regression()"



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

'''
		# hack to try getting around an 'unexpected EOF while parsing' error
		pr = []
		for e in p:
			pr.append(e)
		# pulling out % default for loans the model invests in 
		a = true_results.values
		# same hack as above
		ac = []
		for e in a:
			ac.append(e)
		# finally getting to % default by model
		count_true_true = 0
		#pdb.set_trace()
		for i in range(len(p)):
			if (pr[i] is 1) and (ac[i] is 1):
				count_true_true += 1
		print "percent not-defaulted by model: " + str(float(count_true_true)/sum(p))
		self.display_ROC(p, true_results)
'''
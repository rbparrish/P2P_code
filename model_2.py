import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, ensemble

import pdb

import open_subset as op
import parsers as pa


class Classifier:

	def __init__(self, file_location):
		
		# original dataframe
		self.data = None
		# training dataset
		self.training_dataset = None
		# column with outcomes from original dataframe
		self.predictors = None
		# training feature set
		self.training_features = None
		# predictors from training set
		self.scaler = None
		# scaled training feature set
		self.scaled_training_features = None
		# SKlearn LogRegression classifier
		self.lr_clf = None
		# analysis dataframe
		self.analysis_df = None

		print "pulling data & training logistic regression model"
		self.prep_data(file_location)
		self.scale_data()
		self.train_log_reg()
		self.train_rndm_forest()
		print "ready to build test_results analysis dataframe"
		print "run .create_full_analysis_dataset(<file>)"

	def prep_data(self, file_location):
		# open dataset (NOTE: date for training set is set in 'op')
		self.data = op.open_and_process(file_location, 'transform', 'model2')

		#self.tr_feat = self.data[:n]
		#del self.tr_feat['status_Fully Paid']
		#self.tr_pred = self.pred[:n]

		# separate by issue date (only train using data from before issue date X)
		self.training_data = self.data[self.data['issue_d'] < pa.parse_date("2011-01-01")]
		# delete issue date column (no longer needed)
		del self.training_data['issue_d']
		print self.training_data.columns
		# separate predictors into it's own dataset
		self.predictors = self.training_data['status_in_good_standing']

		# separate features from original training dataset
		self.training_features = self.training_data
		del self.training_features['status_in_good_standing']

	def scale_data(self):
		print "scaling data"
		self.scaler = preprocessing.StandardScaler()
		self.scaled_training_features = self.scaler.fit_transform(self.training_features)

	def train_log_reg(self):
		print "training logistic regression model"
		# train logistic regression model
		self.lr_clf = linear_model.SGDClassifier()
		self.lr_clf.fit(self.scaled_training_features, self.predictors.values)

	def train_rndm_forest(self):

		self.rndm_forest = ensemble.RandomForestClassifier()
		self.rndm_forest.fit(self.scaled_training_features, self.predictors.values)

	def create_full_analysis_dataset(self, file_location):
		##### RUN TESTING DATASET THROUGH PREDICTIVE MODEL #####
		# open dataset (NOTE: date for analysis set is set in 'op')
		self.t_data = op.open_and_process(file_location, 'transform', 'model2')
		print "PREPARING & RUNNING TEST DATASET THROUGH PREDICTIVE MODELS"
		test_data = self.t_data[self.t_data['issue_d'] >= pa.parse_date("2011-01-01")]
		test_data = test_data[test_data['issue_d'] <= "2011-12-31"]
		# delete unneeded columns
		del test_data['status_in_good_standing']
		del test_data['issue_d']
		scaled_test_data = self.scaler.transform(test_data)
		# get lineary regression prediction data (weights)
		lr_prediction_weights = self.lr_clf.decision_function(scaled_test_data)
		# get random forrest prediction data
		rf_predictions = self.rndm_forest.predict(scaled_test_data)

		print "CREATING ANALYSIS DATAFRAME"
		##### LINK PREDICTION WEIGHTS WITH DATA_TABLE USED FOR ANALYSIS #####
		self.a_data = op.open_and_process(file_location, 'transform', 'analysis')
		
		analysis_data = self.a_data
		# only use where index matches those from test_dataset
		analysis_data = analysis_data.loc[test_data.index]
		print "months included in test data: " + str(set(analysis_data['yy-mm_start_date']))

		analysis_data['lin_reg_weights'] = lr_prediction_weights
		analysis_data['rndm_forrest_predictions'] = rf_predictions
		####analysis_data = analysis_data.dropna(axis=0)

		# saving for problem sovling
		print "ANALYSIS COMPLETE"
		self.analysis_df = analysis_data

	def create_predictions(self, threshold):
		pass 





'''
		if prelim_test is "yes":
			if split is "by_grades":
				grades = ['A','B','C','D','E','F','G']
				for letter in grades:
					dt = self.tst_dt[self.tst_dt['grade_'+letter] == 1]
					tst_pred = dt['status_Fully Paid']
					tst_feat = dt
					del tst_feat['status_Fully Paid'] # change to remove using .drop : see http://stackoverflow.com/questions/13411544/delete-column-from-pandas-dataframe
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
		print "None-default rate if fully invest: " + str(float(total_paid)/total_notes)
		# information on performance of the model
		pr = self.clf.predict(features)
		total_investments_by_prediction = sum(pr)
		print "Total number invested by model: " + str(total_investments_by_prediction)
		################
		ac = true_results.values
		correct = 0
		for i in range(len(pr)):
			if (pr[i] == 1) and (ac[i] == 1):
				correct += 1
		print "non-default rate invested by model: " + str(float(correct)/sum(pr))
		self.display_ROC(pr, ac)

	def display_ROC(self, pr, ac):
		##### make ROC graph for the result #####
		if sum(pr) == len(pr):
			print "NO ROC : model invested in all notes"
			print "------------------------------------"
			return None
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

	def single_custom_analysis(self, features):
		prepd_feat = self.scaler.transform(features)
		return self.clf.predict_proba(prepd_feat)
		#predict_scores = predict_proba(prepd_feat)
		#results = []
		#for each in predict_
'''




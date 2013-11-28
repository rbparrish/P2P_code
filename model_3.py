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
		# analysis dataframe (one for each model)
		self.lr_analysis_df = None
		self.rf_analysis_df = None

		print "pulling data & training logistic regression model"
		self.prep_data(file_location)
		self.prep_data_tree(file_location)
		self.scale_data()
		self.train_log_reg()
		self.train_rndm_forest()
		print "ready to build test_results analysis dataframe"
		print ".create_full_analysis_dataset(<file>) for logistic regression"
		print ".create_full_analysis_dataset_tree"

	def prep_data(self, file_location):
		# open dataset (NOTE: date for training set is set in 'op')
		self.data = op.open_and_process(file_location, 'transform', 'model')

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

	def prep_data_tree(self, file_location):
		# open dataset (NOTE: date for training set is set in 'op')
		self.tree_data = op.open_and_process(file_location, 'tree_transform', 'tree_model')

		# separate by issue date (only train using data from before issue date X)
		self.tree_training_data = self.data[self.data['issue_d'] < pa.parse_date("2011-01-01")]
		# delete issue date column (no longer needed)
		del self.tree_training_data['issue_d']
		print self.tree_training_data.columns
		# separate predictors into it's own dataset
		self.tree_predictors = self.tree_training_data['status_in_good_standing']

		# separate features from original training dataset
		self.tree_training_features = self.tree_training_data
		del self.tree_training_features['status_in_good_standing']

	def scale_data(self):
		print "scaling data for Logistic Regression"
		self.scaler = preprocessing.StandardScaler()
		self.scaled_training_features = self.scaler.fit_transform(self.training_features)

	def train_log_reg(self):
		print "Training logistic regression model"
		# train logistic regression model
		self.lr_clf = linear_model.SGDClassifier()
		self.lr_clf.fit(self.scaled_training_features, self.predictors.values)

	def train_rndm_forest(self):
		print "Training Random Forest Model"
		self.rndm_forest = ensemble.RandomForestClassifier()
		self.rndm_forest.fit(self.tree_training_features.values, self.tree_predictors.values)


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
		#analysis_data['rndm_forrest_predictions'] = rf_predictions
		####analysis_data = analysis_data.dropna(axis=0)

		# saving for problem sovling
		print "ANALYSIS COMPLETE"
		self.lr_analysis_df = analysis_data

	def create_full_analysis_dataset_tree(self, file_location):
		##### RUN TESTING DATASET THROUGH PREDICTIVE MODEL #####
		# open dataset (NOTE: date for analysis set is set in 'op')
		self.t_tree_data = op.open_and_process(file_location, 'tree_transform', 'tree_model')
		print "PREPARING & RUNNING TEST DATASET THROUGH RANDOM FOREST PREDICTIVE MODELS"
		test_tree_data = self.t_tree_data[self.t_tree_data['issue_d'] >= pa.parse_date("2011-01-01")]
		test_tree_data = test_tree_data[test_tree_data['issue_d'] <= "2011-12-31"]
		# delete unneeded columns
		del test_tree_data['status_in_good_standing']
		del test_tree_data['issue_d']
		#scaled_test_data = self.scaler.transform(test_data)
		# get lineary regression prediction data (weights)
		rf_prediction_weights = self.rndm_forest.predict_proba(test_tree_data)
		#lr_prediction_weights = self.lr_clf.decision_function(scaled_test_data)
		# get random forrest prediction data
		#rf_predictions = self.rndm_forest.predict(scaled_test_data)

		print "CREATING ANALYSIS DATAFRAME"
		##### LINK PREDICTION WEIGHTS WITH DATA_TABLE USED FOR ANALYSIS #####
		self.tr_data = op.open_and_process(file_location, 'tree_transform', 'analysis')
		
		analysis_data = self.tr_data
		# only use where index matches those from test_dataset
		analysis_data = analysis_data.loc[test_tree_data.index]
		print "months included in test data: " + str(set(analysis_data['yy-mm_start_date']))

		#pdb.set_trace()
		#analysis_data['lin_reg_weights'] = lr_prediction_weights
		prob_default = []
		for each in rf_prediction_weights:
			prob_default.append(float(int(each[0]*100))/100)

		analysis_data['rndm_forrest_predictions'] = prob_default
		####analysis_data = analysis_data.dropna(axis=0)

		# saving for problem sovling
		print "ANALYSIS COMPLETE"
		self.rf_analysis_df = analysis_data

	def create_predictions(self, threshold):
		pass 


